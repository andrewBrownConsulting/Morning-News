"""Daily news digest: fetch RSS feeds, let an LLM pick what Andy would enjoy, email the top stories.

Usage:
    python3 digest.py            # fetch, score, send email
    python3 digest.py --dry-run  # fetch, score, write digest.html instead of emailing
"""

import html
import json
import os
import smtplib
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from email.utils import parsedate_to_datetime
from pathlib import Path

HERE = Path(__file__).parent

FEEDS = {
    "BBC": "https://feeds.bbci.co.uk/news/rss.xml",
    "Guardian": "https://www.theguardian.com/uk/rss",
    "Hacker News": "https://news.ycombinator.com/rss",
    "Ars Technica": "https://feeds.arstechnica.com/arstechnica/index",
    "The Verge": "https://www.theverge.com/rss/index.xml",
    "Hackaday": "https://hackaday.com/blog/feed/",
}

TOP_N = 10
MAX_AGE = timedelta(hours=24)
BATCH_SIZE = 50
# Any OpenAI-compatible chat completions API works; defaults to Google Gemini's free tier
API_URL = os.environ.get("API_URL", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")
MODEL = os.environ.get("MODEL", "gemini-3.8-flash")
FALLBACK_MODEL = os.environ.get("FALLBACK_MODEL", "gemini-3.5-flash-lite")
RETRY_DELAYS = [10, 30]  # seconds to wait before each retry on overload/timeout
RETRYABLE_HTTP = {429, 500, 502, 503, 504}
ATOM = "{http://www.w3.org/2005/Atom}"


def load_dotenv():
    """Read KEY=value lines from .env (local runs only; GitHub Actions uses secrets)."""
    env_file = HERE / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def http_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "news-digest/0.1"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read()


def parse_date(text):
    if not text:
        return None
    try:
        return parsedate_to_datetime(text)  # RSS: "Tue, 21 Oct 2026 07:00:00 GMT"
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))  # Atom: ISO 8601
    except ValueError:
        return None


def fetch_feed(source, url):
    root = ET.fromstring(http_get(url))
    stories = []
    for item in root.iter("item"):  # RSS
        stories.append({
            "source": source,
            "title": (item.findtext("title") or "").strip(),
            "link": (item.findtext("link") or "").strip(),
            "date": parse_date(item.findtext("pubDate")),
        })
    for entry in root.iter(f"{ATOM}entry"):  # Atom (The Verge)
        link = entry.find(f"{ATOM}link")
        stories.append({
            "source": source,
            "title": (entry.findtext(f"{ATOM}title") or "").strip(),
            "link": link.get("href", "") if link is not None else "",
            "date": parse_date(entry.findtext(f"{ATOM}published") or entry.findtext(f"{ATOM}updated")),
        })
    return stories


def fetch_all():
    cutoff = datetime.now(timezone.utc) - MAX_AGE
    stories, seen = [], set()
    for source, url in FEEDS.items():
        try:
            items = fetch_feed(source, url)
        except Exception as e:
            print(f"  ! {source} failed: {e}")
            continue
        kept = 0
        for s in items:
            key = s["title"].lower()
            if not s["title"] or not s["link"] or key in seen:
                continue
            if s["date"] and s["date"].tzinfo and s["date"] < cutoff:
                continue  # stories with no date are kept (e.g. Hacker News)
            seen.add(key)
            stories.append(s)
            kept += 1
        print(f"  {source}: {kept} stories")
    return stories


class Retryable(Exception):
    """A temporary API failure (overloaded, rate-limited, timed out) worth retrying."""


def score_batch(stories, profile, api_key, model):
    """Ask the LLM to score each headline 0-10 for how much Andy would enjoy it."""
    headlines = "\n".join(f"{i}. [{s['source']}] {s['title']}" for i, s in enumerate(stories))
    prompt = (
        f"Here is a reader's taste profile:\n\n{profile}\n\n"
        f"Score each headline below from 0 to 10 for how much this reader would ENJOY reading it. "
        f"Give a short reason (max 12 words) addressed to the reader as 'you'.\n\n{headlines}\n\n"
        'Reply with JSON only: {"scores": [{"id": 0, "score": 7, "reason": "..."}, ...]}'
    )
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }).encode()
    req = urllib.request.Request(API_URL, data=body, headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "news-digest/0.1",
    })
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        error = Retryable if e.code in RETRYABLE_HTTP else ValueError
        raise error(f"HTTP {e.code}: {e.read()[:300]!r}") from None
    except (TimeoutError, urllib.error.URLError) as e:
        raise Retryable(f"network: {e}") from None
    try:
        content = json.loads(raw)["choices"][0]["message"]["content"] or ""
        # Models sometimes wrap JSON in ```json fences or add text around it
        result = json.loads(content[content.index("{"):content.rindex("}") + 1])
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise ValueError(f"{e}; raw response: {raw[:500]!r}") from None
    for entry in result.get("scores", []):
        i = entry.get("id")
        if isinstance(i, int) and 0 <= i < len(stories):
            stories[i]["score"] = entry.get("score", 0)
            stories[i]["reason"] = entry.get("reason", "")


def score_with_retries(stories, profile, api_key):
    """Try the main model with retries, then the fallback model, before giving up."""
    for model in (MODEL, FALLBACK_MODEL):
        for delay in RETRY_DELAYS + [None]:
            try:
                score_batch(stories, profile, api_key, model)
                return
            except Retryable as e:
                print(f"    {model}: {str(e)[:80]}")
                if delay is None:
                    break
                print(f"    retrying in {delay}s...")
                time.sleep(delay)
    raise RuntimeError("all models and retries failed")


def score_all(stories, profile, api_key):
    for start in range(0, len(stories), BATCH_SIZE):
        batch = stories[start:start + BATCH_SIZE]
        try:
            score_with_retries(batch, profile, api_key)
        except Exception as e:
            print(f"  ! scoring batch {start // BATCH_SIZE + 1} failed: {e}")
    scored = [s for s in stories if "score" in s]
    return sorted(scored, key=lambda s: s["score"], reverse=True)[:TOP_N]


def build_html(picks, apology=None):
    today = datetime.now().strftime("%A %-d %B")
    rows = "".join(
        f"""<tr><td style="padding:12px 0;border-bottom:1px solid #eee">
          <a href="{html.escape(s['link'])}" style="font-size:16px;color:#1a0dab;text-decoration:none;font-weight:600">{html.escape(s['title'])}</a>
          <div style="font-size:13px;color:#666;margin-top:4px">{html.escape(" · ".join(x for x in (s['source'], f"{s['score']}/10" if 'score' in s else '', s.get('reason', '')) if x))}</div>
        </td></tr>"""
        for s in picks
    )
    return f"""<html><body style="font-family:-apple-system,Segoe UI,sans-serif;max-width:600px;margin:auto;padding:16px">
      <h2 style="margin-bottom:4px">Your news, {today}</h2>
      {f'<p style="background:#fff4e5;border-left:4px solid #f0a020;padding:10px 12px">{html.escape(apology)}</p>' if apology else f'<p style="color:#666;margin-top:0">{len(picks)} stories picked for you</p>'}
      <table style="width:100%;border-collapse:collapse">{rows}</table>
    </body></html>"""


def send_email(subject, body_html):
    user = os.environ["GMAIL_ADDRESS"]
    msg = MIMEText(body_html, "html")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = os.environ.get("DIGEST_TO", user)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(user, os.environ["GMAIL_APP_PASSWORD"].replace(" ", ""))
        smtp.send_message(msg)


def main():
    load_dotenv()
    dry_run = "--dry-run" in sys.argv
    api_key = os.environ.get("API_KEY")
    if not api_key:
        sys.exit("API_KEY is not set (add it to .env or GitHub secrets)")

    print("Fetching feeds...")
    stories = fetch_all()
    print(f"Scoring {len(stories)} stories with {MODEL}...")
    picks = score_all(stories, (HERE / "profile.txt").read_text(), api_key)
    apology = None
    if picks:
        subject = f"Your news: {picks[0]['title']}"
    else:
        # Failsafe: the AI is down, so send the BBC headlines unfiltered
        print("No stories were scored; falling back to BBC headlines")
        picks = [s for s in stories if s["source"] == "BBC"][:TOP_N]
        apology = ("Sorry, the AI isn't AIing today, so I couldn't pick stories for you. "
                   "Here are the top BBC headlines instead.")
        subject = "Your news (AI day off): BBC headlines"
    if not picks:
        sys.exit("No stories were fetched, so there's nothing to send")

    body = build_html(picks, apology)
    if dry_run:
        out = HERE / "digest.html"
        out.write_text(body)
        print(f"Dry run: wrote {out}")
    else:
        send_email(subject, body)
        print("Email sent")


if __name__ == "__main__":
    main()
