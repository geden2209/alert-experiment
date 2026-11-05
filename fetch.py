import csv
import os
import sys
from datetime import datetime, timezone

import requests

TOKEN = os.environ.get("ALERTS_TOKEN")
# Название области точно как в API, например: "Київська область", "м. Київ"
OBLAST = os.environ.get("OBLAST", "Київська область")

URL = "https://api.alerts.in.ua/v1/alerts/active.json"
OUT = "data/snapshots.csv"
HEADER = [
    "timestamp_utc",
    "oblast",
    "air_raid_in_oblast",
    "alerts_in_oblast",
    "alerts_total",
    "oblasts_with_air_raid",
]


def main():
    if not TOKEN:
        sys.exit("Не задан секрет ALERTS_TOKEN")

    resp = requests.get(
        URL,
        headers={"Authorization": f"Bearer {TOKEN}"},
        timeout=30,
    )
    resp.raise_for_status()
    alerts = resp.json().get("alerts", [])

    in_oblast = [a for a in alerts if a.get("location_oblast") == OBLAST]
    air_in_oblast = [a for a in in_oblast if a.get("alert_type") == "air_raid"]
    air_oblasts = sorted(
        {
            a.get("location_oblast", "")
            for a in alerts
            if a.get("alert_type") == "air_raid"
        }
    )

    row = [
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        OBLAST,
        1 if air_in_oblast else 0,
        len(in_oblast),
        len(alerts),
        ";".join(air_oblasts),
    ]

    os.makedirs("data", exist_ok=True)
    is_new = not os.path.exists(OUT)
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(HEADER)
        writer.writerow(row)

    print("Записано:", row)


if __name__ == "__main__":
    main()
