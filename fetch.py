import os
import sys

import requests

REPO = "Vadimkin/ukrainian-air-raid-sirens-dataset"
PATH = "datasets/official_data_en.csv"
BRANCHES = ["main", "master"]
OUT = "data/official_data_en.csv"


def download():
    last_error = ""
    for branch in BRANCHES:
        url = f"https://raw.githubusercontent.com/{REPO}/{branch}/{PATH}"
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200:
            return url, resp.content
        last_error = f"{url} -> HTTP {resp.status_code}"
    sys.exit("Не удалось скачать файл: " + last_error)


def main():
    url, content = download()

    os.makedirs("data", exist_ok=True)
    with open(OUT, "wb") as f:
        f.write(content)

    lines = content.decode("utf-8", errors="replace").splitlines()
    print("Источник:", url)
    print("Строк в файле:", len(lines))
    print("Колонки:", lines[0] if lines else "(пусто)")
    for line in lines[1:4]:
        print(line)


if __name__ == "__main__":
    main()
