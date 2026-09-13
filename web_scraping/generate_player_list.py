import csv

INPUT_FILE = "matches_optimized.csv"
OUTPUT_FILE = "players.csv"
LAST_N = 5

players = set()

with open(INPUT_FILE, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if int(row["date"][:4]) < 2010:
            continue
        if "Friendl" in row["competition"]:
            continue
        for col in ("home_players", "away_players"):
            all_players = [p.strip() for p in row[col].split("|") if p.strip()]
            for name in all_players[-LAST_N:]:
                players.add(name)

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["player"])
    for name in sorted(players):
        writer.writerow([name])

print(f"{len(players)} distinct players written to {OUTPUT_FILE}")