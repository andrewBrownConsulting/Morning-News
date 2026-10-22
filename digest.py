"""Daily morning briefing: calendar, weather, and news an LLM thinks Andy would enjoy, by email.

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
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import caldav
import icalendar
import recurring_ical_events

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
TZ = ZoneInfo("Europe/London")
WEATHER_LOCATION = os.environ.get("WEATHER_LOCATION", "Hammersmith")
CALDAV_URL = os.environ.get("CALDAV_URL", "https://caldav.icloud.com/")
WRITE_CALENDAR = "Morning Briefing"
WEATHER_CODES = {
    0: "☀️ Clear", 1: "🌤️ Mostly clear", 2: "⛅ Partly cloudy", 3: "☁️ Overcast",
    45: "🌫️ Fog", 48: "🌫️ Fog", 51: "🌦️ Light drizzle", 53: "🌦️ Drizzle", 55: "🌦️ Heavy drizzle",
    56: "🌧️ Freezing drizzle", 57: "🌧️ Freezing drizzle", 61: "🌧️ Light rain", 63: "🌧️ Rain",
    65: "🌧️ Heavy rain", 66: "🌧️ Freezing rain", 67: "🌧️ Freezing rain", 71: "🌨️ Light snow",
    73: "🌨️ Snow", 75: "🌨️ Heavy snow", 77: "🌨️ Snow grains", 80: "🌦️ Showers", 81: "🌦️ Showers",
    82: "⛈️ Heavy showers", 85: "🌨️ Snow showers", 86: "🌨️ Snow showers", 95: "⛈️ Thunderstorm",
    96: "⛈️ Thunderstorm with hail", 99: "⛈️ Thunderstorm with hail",
}


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


def caldav_principal():
    """Log in to iCloud over CalDAV (private, read/write). Returns None if not configured."""
    user, password = os.environ.get("ICLOUD_USERNAME"), os.environ.get("ICLOUD_APP_PASSWORD")
    if not (user and password):
        return None
    client = caldav.DAVClient(url=CALDAV_URL, username=user, password=password.replace(" ", ""))
    return client.principal()


def load_calendar():
    """All events as one icalendar.Calendar: via CalDAV if configured, else the ICAL_URL link."""
    principal = caldav_principal()
    if principal is None:
        url = os.environ.get("ICAL_URL", "").replace("webcal://", "https://")
        return icalendar.Calendar.from_ical(http_get(url)) if url else None
    merged = icalendar.Calendar()
    start = datetime.now(TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    for cal in principal.calendars():
        try:
            # Recurring masters are returned if any occurrence overlaps; we expand them below
            found = cal.search(start=start, end=start + timedelta(days=2), event=True)
        except Exception as e:
            print(f"  ! calendar {cal.get_display_name()} skipped: {e}")
            continue
        for obj in found:
            for component in icalendar.Calendar.from_ical(obj.data).walk():
                if component.name in ("VEVENT", "VTIMEZONE"):
                    merged.add_component(component)
    return merged


def fetch_calendar():
    """Today's and tomorrow's events, as {date: [(time label, title), ...]}."""
    cal = load_calendar()
    if cal is None:
        return None
    today = datetime.now(TZ).date()
    days = {}
    for day in (today, today + timedelta(days=1)):
        events = []
        for ev in recurring_ical_events.of(cal).at(day):
            start = ev["DTSTART"].dt
            if isinstance(start, datetime):
                if start.tzinfo:
                    start = start.astimezone(TZ)
                if start.date() < day:
                    label = "(cont.)"  # started on an earlier day
                else:
                    label = start.strftime("%H:%M")
            else:
                label = "All day"
            events.append((label, str(ev.get("SUMMARY", "(no title)"))))
        # All-day and continuing events first, then by time
        days[day] = sorted(events, key=lambda e: (e[0][0].isdigit(), e[0]))
    return days


def briefing_calendar(principal):
    """The app's own calendar for events it creates, made on first use so they're easy to hide."""
    for cal in principal.calendars():
        if cal.get_display_name() == WRITE_CALENDAR:
            return cal
    return principal.make_calendar(name=WRITE_CALENDAR)


def add_event(summary, start, end, description="", uid=None):
    """Create (or update, if uid already exists) an event in the app's calendar.

    Pass a stable uid (e.g. "climbing-2026-10-23") so re-running the script doesn't duplicate events.
    start/end are timezone-aware datetimes, or dates for an all-day event.
    """
    principal = caldav_principal()
    if principal is None:
        raise RuntimeError("Writing events needs ICLOUD_USERNAME and ICLOUD_APP_PASSWORD")
    event = icalendar.Event()
    event.add("uid", uid or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}@morning-news")
    event.add("dtstamp", datetime.now(timezone.utc))
    event.add("dtstart", start)
    event.add("dtend", end)
    event.add("summary", summary)
    if description:
        event.add("description", description)
    cal = icalendar.Calendar()
    cal.add("prodid", "-//Morning News//EN")
    cal.add("version", "2.0")
    cal.add_component(event)
    return briefing_calendar(principal).add_event(cal.to_ical().decode())


def fetch_weather():
    """Today's forecast for WEATHER_LOCATION from Open-Meteo (free, no key)."""
    geo = json.loads(http_get("https://geocoding-api.open-meteo.com/v1/search?"
                              + urlencode({"name": WEATHER_LOCATION, "count": 1})))
    place = geo["results"][0]
    params = urlencode({
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,"
                 "precipitation_probability_max,wind_speed_10m_max",
        "timezone": "Europe/London",
        "past_days": 1,
        "forecast_days": 1,
    })
    d = json.loads(http_get(f"https://api.open-meteo.com/v1/forecast?{params}"))["daily"]
    # Index 0 is yesterday (for rock dryness), index 1 is today
    rain_yesterday, rain_today = d["precipitation_sum"][0] or 0, d["precipitation_sum"][1] or 0
    return {
        "place": place["name"],
        "summary": WEATHER_CODES.get(d["weather_code"][1], "🌡️ Mixed"),
        "high": round(d["temperature_2m_max"][1]),
        "low": round(d["temperature_2m_min"][1]),
        "rain_chance": d["precipitation_probability_max"][1] or 0,
        "rain_mm": rain_today,
        "wind": round(d["wind_speed_10m_max"][1]),
        "rock_dry": rain_yesterday < 1 and rain_today < 1,
    }


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


def calendar_html(days):
    if days is None:
        return ""
    parts = []
    for day, events in days.items():
        label = "Today" if day == datetime.now(TZ).date() else "Tomorrow"
        if events:
            lines = "".join(
                f'<div style="padding:2px 0"><span style="color:#666;display:inline-block;width:64px">{html.escape(t)}</span>{html.escape(title)}</div>'
                for t, title in events)
        else:
            lines = '<div style="color:#666">Nothing planned</div>'
        parts.append(f'<div style="margin-bottom:8px"><b>{label}</b>{lines}</div>')
    return f'<h3 style="margin-bottom:6px">📅 Calendar</h3>{"".join(parts)}'


def weather_html(w):
    if w is None:
        return ""
    rock = "🧗 Rock should be dry" if w["rock_dry"] else "🧗 Rock likely wet, so maybe an indoor day"
    return f"""<h3 style="margin-bottom:6px">Weather in {html.escape(w['place'])}</h3>
      <div style="font-size:18px">{w['summary']} · {w['high']}° / {w['low']}°</div>
      <div style="color:#666;font-size:14px;margin-top:2px">Rain {w['rain_chance']}% ({w['rain_mm']:.1f} mm) · Wind up to {w['wind']} km/h</div>
      <div style="font-size:14px;margin-top:4px">{rock}</div>"""


def build_html(picks, apology=None, calendar=None, weather=None):
    today = datetime.now(TZ).strftime("%A %-d %B")
    rows = "".join(
        f"""<tr><td style="padding:12px 0;border-bottom:1px solid #eee">
          <a href="{html.escape(s['link'])}" style="font-size:16px;color:#1a0dab;text-decoration:none;font-weight:600">{html.escape(s['title'])}</a>
          <div style="font-size:13px;color:#666;margin-top:4px">{html.escape(" · ".join(x for x in (s['source'], f"{s['score']}/10" if 'score' in s else '', s.get('reason', '')) if x))}</div>
        </td></tr>"""
        for s in picks
    )
    return f"""<html><body style="font-family:-apple-system,Segoe UI,sans-serif;max-width:600px;margin:auto;padding:16px">
      <h2 style="margin-bottom:4px">Good morning, Andy</h2>
      <p style="color:#666;margin-top:0">{today}</p>
      {calendar_html(calendar)}
      {weather_html(weather)}
      <h3 style="margin-bottom:6px">📰 News</h3>
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

    calendar = weather = None
    try:
        calendar = fetch_calendar()
        print("Calendar: " + ("skipped (not configured)" if calendar is None else
                              ", ".join(f"{len(e)} events" for e in calendar.values())))
    except Exception as e:
        print(f"  ! calendar failed: {e}")
    try:
        weather = fetch_weather()
        print(f"Weather: {weather['place']} {weather['summary']}")
    except Exception as e:
        print(f"  ! weather failed: {e}")

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

    body = build_html(picks, apology, calendar, weather)
    if dry_run:
        out = HERE / "digest.html"
        out.write_text(body)
        print(f"Dry run: wrote {out}")
    else:
        send_email(subject, body)
        print("Email sent")


if __name__ == "__main__":
    main()
