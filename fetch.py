import os
import sys

import requests

REPO = "Vadimkin/ukrainian-air-raid-sirens-dataset"
BRANCHES = ["main", "master"]
NAME = "volunteer_data_en.csv"
OUT = f"data/{NAME}"


def download():
    errors = []
    for branch in BRANCHES:
        url = f"https://raw.githubusercontent.com/{REPO}/{branch}/datasets/{NAME}"
        try:
            resp = requests.get(url, timeout=120)
        except requests.RequestException as exc:
            errors.append(f"{url} -> {exc}")
            continue
        if resp.status_code == 200:
            return url, resp.content
        errors.append(f"{url} -> HTTP {resp.status_code}")
    sys.exit("Не удалось скачать файл:\n" + "\n".join(errors))


def main():
    url, content = download()
    os.makedirs("data", exist_ok=True)
    with open(OUT, "wb") as f:
        f.write(content)

    lines = content.decode("utf-8", errors="replace").splitlines()
    print("Источник:", url)
    print("Строк в файле:", len(lines))
    print("Колонки:", lines[0] if lines else "(пусто)")
    print("Последняя строка:", lines[-1] if lines else "(пусто)")


if __name__ == "__main__":
    main()
