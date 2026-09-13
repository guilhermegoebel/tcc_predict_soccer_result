import csv
import time
import logging
import re
import threading
from pathlib import Path

import requests
from bs4 import BeautifulSoup

START_ID = 5581
END_ID = 47000
OUTPUT_FILE = "matches_optimized.csv"
NUM_THREADS = 4
MAX_RETRIES = 3
RETRY_DELAY = 10
MIN_YEAR = 2000
BASE_URL = "https://www.national-football-teams.com/matches/report/{}/a.html"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [T%(thread)d] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.national-football-teams.com/",
}

CSV_FIELDNAMES = [
    "match_id",
    "date",
    "home_team",
    "away_team",
    "score",
    "score_regular",
    "score_penalties",
    "competition",
    "home_players",
    "away_players",
]

write_lock = threading.Lock()


def fetch_page(match_id):
    url = BASE_URL.format(match_id)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            if resp.status_code == 404:
                return None, 404
            if resp.status_code == 429:
                log.warning("Rate limited on id=%d, sleeping %ds", match_id, RETRY_DELAY * attempt)
                time.sleep(RETRY_DELAY * attempt)
                continue
            if resp.status_code != 200:
                log.warning("HTTP %d for id=%d", resp.status_code, match_id)
                time.sleep(RETRY_DELAY)
                continue
            return resp.text, resp.status_code
        except requests.RequestException as exc:
            log.warning("Request error id=%d attempt=%d: %s", match_id, attempt, exc)
            time.sleep(RETRY_DELAY)
    return None, -1


def parse_match(html, match_id):
    soup = BeautifulSoup(html, "html.parser")

    h1 = soup.find("h1")
    if not h1:
        return None
    small = h1.find("small")
    if not small:
        return None
    date_str = small.get_text(strip=True)
    try:
        year = int(date_str[:4])
    except ValueError:
        return None
    if year < MIN_YEAR:
        return None

    score_section = soup.find("section", class_="section-center")
    if not score_section:
        return None
    score_h2 = score_section.find("h2")
    if not score_h2:
        return None

    flag_imgs = score_h2.find_all("img")
    if len(flag_imgs) < 2:
        return None
    home_team = flag_imgs[0].get("title", "").strip()
    away_team = flag_imgs[1].get("title", "").strip()

    score_regular = None
    score_penalties = None
    score_display = None

    score_span = score_h2.find("span", title=lambda t: t and "regular time" in t)
    if score_span:
        score_regular = score_span.get_text(strip=True)
        pen_span = score_h2.find("span", title=lambda t: t and "Penalty" in t)
        score_penalties = pen_span.get_text(strip=True) if pen_span else None
        score_display = score_regular
        if score_penalties:
            score_display = f"{score_regular} (pen {score_penalties})"
    else:
        raw_text = score_h2.get_text(separator=" ", strip=True)
        raw_text = re.sub(r"\s+", " ", raw_text)
        raw_text = raw_text.replace(home_team, "").replace(away_team, "").strip()
        score_display = raw_text if raw_text else "?"

    breadcrumb = soup.find("ul", class_="breadcrumb")
    competition_parts = []
    if breadcrumb:
        items = breadcrumb.find_all("li")
        collecting = False
        for item in items:
            span = item.find("span", itemprop="title")
            if not span:
                continue
            text = span.get_text(strip=True)
            if text == "Matches":
                collecting = True
                continue
            if not collecting:
                continue
            if re.match(r"^\d{4}$", text):
                continue
            if item == items[-1]:
                break
            competition_parts.append(text)
    competition = " / ".join(competition_parts) if competition_parts else "Unknown"

    team_sections = []
    for team_div in [soup.find("div", id="home_team"), soup.find("div", id="away_team")]:
        if not team_div:
            team_sections.append([])
            continue
        lineup_heading = team_div.find("h6", string=lambda s: s and "Starting Line-Up" in s)
        if not lineup_heading:
            team_sections.append([])
            continue
        lineup_div = lineup_heading.find_parent("div", class_="heading")
        if not lineup_div:
            team_sections.append([])
            continue
        players_div = lineup_div.find_next_sibling("div", class_="players")
        if not players_div:
            team_sections.append([])
            continue
        player_divs = players_div.find_all("div", attrs={"itemprop": "athletes"})
        names = []
        for p in player_divs[:11]:
            family = p.find("span", itemprop="familyName")
            given = p.find("span", itemprop="givenName")
            if family and given:
                names.append(f"{given.get_text(strip=True)} {family.get_text(strip=True)}")
            elif family:
                names.append(family.get_text(strip=True))
            elif given:
                names.append(given.get_text(strip=True))
        team_sections.append(names)

    home_players = team_sections[0] if len(team_sections) > 0 else []
    away_players = team_sections[1] if len(team_sections) > 1 else []

    return {
        "match_id": match_id,
        "date": date_str,
        "home_team": home_team,
        "away_team": away_team,
        "score": score_display,
        "score_regular": score_regular or "",
        "score_penalties": score_penalties or "",
        "competition": competition,
        "home_players": "|".join(home_players),
        "away_players": "|".join(away_players),
    }


def get_processed_ids(output_file):
    processed = set()
    if not Path(output_file).exists():
        return processed
    with open(output_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                processed.add(int(row["match_id"]))
            except (KeyError, ValueError):
                pass
    return processed


def worker(match_ids, writer, csvfile):
    for match_id in match_ids:
        log.info("Fetching id=%d", match_id)
        html, status = fetch_page(match_id)

        if status == 404 or html is None:
            log.info("No page for id=%d (status=%s)", match_id, status)
            continue

        try:
            result = parse_match(html, match_id)
        except Exception as exc:
            log.error("Parse error id=%d: %s", match_id, exc)
            result = None

        if result is None:
            log.info("Skipped id=%d (no data or pre-2000)", match_id)
        else:
            with write_lock:
                writer.writerow(result)
                csvfile.flush()
            log.info(
                "Saved id=%d | %s | %s vs %s | %s | %s",
                match_id,
                result["date"],
                result["home_team"],
                result["away_team"],
                result["score"],
                result["competition"],
            )


def main():
    processed_ids = get_processed_ids(OUTPUT_FILE)
    file_exists = Path(OUTPUT_FILE).exists()

    all_ids = [i for i in range(START_ID, END_ID + 1) if i not in processed_ids]
    log.info("Total IDs to process: %d", len(all_ids))

    chunks = [all_ids[i::NUM_THREADS] for i in range(NUM_THREADS)]

    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        threads = []
        for chunk in chunks:
            t = threading.Thread(target=worker, args=(chunk, writer, csvfile))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

    log.info("Done. Results in %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()