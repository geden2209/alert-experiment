import os
import sys

import requests

REPO = "Vadimkin/ukrainian-air-raid-sirens-dataset"
BRANCHES = ["main", "master"]
FILES = {
    "official_data_en.csv": True,    # обязательный
    "volunteer_data_en.csv": False,  # необязательный, смотрим что внутри
}


def download(name):
    errors = []
    for branch in BRANCHES:
        url = f"https://raw.githubusercontent.com/{REPO}/{branch}/datasets/{name}"
        try:
            resp = requests.get(url, timeout=60)
        except requests.RequestException as exc:
            errors.append(f"{url} -> {exc}")
            continue
        if resp.status_code == 200:
            return url, resp.content
        errors.append(f"{url} -> HTTP {resp.status_code}")
    return None, "\n".join(errors)


def main():
    os.makedirs("data", exist_ok=True)
    for name, required in FILES.items():
        url, content = download(name)
        print(f"=== {name} ===")
        if url is None:
            print("Не удалось скачать:\n" + content)
            if required:
                sys.exit(1)
            continue

        with open(f"data/{name}", "wb") as f:
            f.write(content)

        lines = content.decode("utf-8", errors="replace").splitlines()
        print("Источник:", url)
        print("Строк в файле:", len(lines))
        print("Колонки:", lines[0] if lines else "(пусто)")
        print("Первые строки:")
        for line in lines[1:4]:
            print("  ", line)
        print("Последние строки:")
        for line in lines[-3:]:
            print("  ", line)


if __name__ == "__main__":
    main()
