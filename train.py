import html
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score

CSV = "data/official_data_en.csv"
OBLAST = "Dnipropetrovska oblast"
# Соседние области (названия как в датасете; отсутствующие пропускаются)
NEIGHBORS = [
    "Zaporizka oblast",
    "Donetska oblast",
    "Kharkivska oblast",
    "Poltavska oblast",
    "Kirovohradska oblast",
    "Mykolaivska oblast",
    "Khersonska oblast",
]
TEST_SHARE = 0.2
# Какие уровни тревог считать. Громадские тревоги на востоке почти не прекращаются,
# поэтому в ряд берём только область и районы.
LEVELS = ["oblast", "raion"]


def active_series(df, index):
    """1, если в этот час шла хоть одна тревога (любого уровня) в df."""
    n = len(index)
    if len(df) == 0:
        return np.zeros(n, dtype=int)
    start = ((df["started_at"].dt.floor("h") - index[0]) // pd.Timedelta("1h")).to_numpy()
    end = ((df["finished_at"].dt.floor("h") - index[0]) // pd.Timedelta("1h")).to_numpy()
    start = np.clip(start, 0, n - 1)
    end = np.clip(end, 0, n - 1)
    diff = np.zeros(n + 1, dtype=int)
    np.add.at(diff, start, 1)
    np.add.at(diff, end + 1, -1)
    return (np.cumsum(diff[:n]) > 0).astype(int)


def main():
    df = pd.read_csv(CSV)
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True, format="ISO8601")
    df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True, format="ISO8601")
    last_known = max(df["started_at"].max(), df["finished_at"].max())

    # Тревоги без времени окончания: недавние считаем идущими, старые - 30 минут
    no_end = df["finished_at"].isna()
    recent = df["started_at"] > last_known - pd.Timedelta("3h")
    print("Строк без finished_at:", int(no_end.sum()), "из них старых:", int((no_end & ~recent).sum()))
    old_no_end = no_end & ~recent
    df.loc[old_no_end, "finished_at"] = df.loc[old_no_end, "started_at"] + pd.Timedelta("30min")
    df["finished_at"] = df["finished_at"].fillna(last_known)
    df = df[df["finished_at"] >= df["started_at"]]

    index = pd.date_range(
        df["started_at"].min().floor("h"), last_known.floor("h"), freq="h", tz="UTC"
    )

    # Диагностика: доля тревожных часов по уровням и годам для нашей области
    own = df[df["oblast"] == OBLAST]
    print("Доля тревожных часов по уровням и годам:")
    for level in sorted(df["level"].dropna().unique()):
        ser = pd.Series(active_series(own[own["level"] == level], index), index=index)
        by_year = ser.groupby(ser.index.year).mean().round(3)
        print(f"  {level}:", " ".join(f"{y}={v}" for y, v in by_year.items()))

    df = df[df["level"].isin(LEVELS)]
    by_oblast = {name: active_series(g, index) for name, g in df.groupby("oblast")}
    if OBLAST not in by_oblast:
        raise SystemExit(f"Нет области {OBLAST}. Есть: {sorted(by_oblast)}")

    missing = [n for n in NEIGHBORS if n not in by_oblast]
    if missing:
        print("Нет соседей в данных:", missing)
    neighbors = [n for n in NEIGHBORS if n in by_oblast]

    s = pd.Series(by_oblast[OBLAST], index=index)

    # Диагностика
    mine = df[df["oblast"] == OBLAST].copy()
    mine["hours"] = (mine["finished_at"] - mine["started_at"]).dt.total_seconds() / 3600
    print("Самые длинные тревоги по области:")
    print(mine.sort_values("hours", ascending=False).head(5)[["level", "started_at", "finished_at", "hours"]].to_string())
    print("Доля тревожных часов по годам:")
    print(s.groupby(s.index.year).mean().round(3).to_string())
    X = pd.DataFrame(index=index)
    X["hour"] = index.hour
    X["dow"] = index.day_of_week
    X["month"] = index.month
    X["now"] = s
    for lag in (1, 2, 3):
        X[f"lag{lag}"] = s.shift(lag)
    for w in (3, 6, 24, 168):
        X[f"sum{w}h"] = s.rolling(w, min_periods=1).sum()

    idx = np.arange(len(s))
    last = np.maximum.accumulate(np.where(s.to_numpy() == 1, idx, -1))
    X["hours_since"] = np.minimum(np.where(last >= 0, idx - last, 168), 168)

    X["neigh_now"] = sum(by_oblast[n] for n in neighbors) if neighbors else 0
    X["national_now"] = sum(by_oblast.values())

    y = s.shift(-1)
    live_row = X.iloc[[-1]]  # прогноз на час после последней записи датасета
    data = X.iloc[:-1].copy()
    target = y.iloc[:-1].astype(int)

    split = int(len(data) * (1 - TEST_SHARE))
    X_tr, X_te = data.iloc[:split], data.iloc[split:]
    y_tr, y_te = target.iloc[:split], target.iloc[split:]

    share = float(y_te.mean())
    if share > 0.6 or share < 0.001:
        raise SystemExit(
            f"Подозрительные данные: доля тревожных часов в тесте {share:.3f}. Смотрите диагностику выше."
        )

    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=42
    )
    model.fit(X_tr, y_tr)
    p_te = model.predict_proba(X_te)[:, 1]

    base_now = X_te["now"].to_numpy()
    base_yday = s.shift(23).iloc[:-1].iloc[split:].fillna(0).to_numpy()

    def metrics(score):
        return roc_auc_score(y_te, score), average_precision_score(y_te, score)

    rows = [
        ("Бустинг", *metrics(p_te)),
        ("База: как сейчас", *metrics(base_now)),
        ("База: как вчера в это время", *metrics(base_yday)),
    ]

    imp = permutation_importance(
        model, X_te, y_te, scoring="average_precision", n_repeats=3, random_state=42
    )
    importance = sorted(zip(X.columns, imp.importances_mean), key=lambda t: -t[1])

    prob = float(model.predict_proba(live_row)[0, 1])
    data_end = index[-1].strftime("%Y-%m-%d %H:%M UTC")
    now_dt = datetime.now(timezone.utc)
    now_utc = now_dt.strftime("%Y-%m-%d %H:%M UTC")
    stale_days = (now_dt - index[-1].to_pydatetime()).days
    stale = (
        f'<div class="warn">Данные в датасете закончились {stale_days} дн. назад. '
        "Прогноз относится к тому моменту, а не к текущему.</div>"
        if stale_days >= 3
        else ""
    )

    print(f"Строк: {len(data)}, тест: {len(X_te)}, доля тревожных часов в тесте: {share:.3f}")
    for name, auc, ap in rows:
        print(f"{name}: ROC-AUC={auc:.3f}, PR-AUC={ap:.3f}")
    print("Вероятность на следующий час после конца данных:", round(prob, 3))

    os.makedirs("docs", exist_ok=True)
    table = "".join(
        f"<tr><td>{html.escape(n)}</td><td>{a:.3f}</td><td>{p:.3f}</td></tr>"
        for n, a, p in rows
    )
    imp_rows = "".join(
        f"<tr><td>{html.escape(n)}</td><td>{v:.4f}</td></tr>" for n, v in importance[:8]
    )
    page = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Эксперимент: тревоги</title>
<style>
:root{{color-scheme:light dark;--bg:#fff;--fg:#111;--mut:#666;--line:#ddd}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111;--fg:#eee;--mut:#999;--line:#333}}}}
body{{font-family:system-ui,sans-serif;background:var(--bg);color:var(--fg);margin:0;padding:16px;max-width:560px}}
h1{{font-size:20px}} h2{{font-size:16px;margin-top:24px}}
.big{{font-size:44px;font-weight:700}} .mut{{color:var(--mut);font-size:13px}}
.warn{{border:1px solid #c80;border-radius:8px;padding:8px;margin:8px 0;font-size:13px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{border-bottom:1px solid var(--line);padding:6px 4px;text-align:left}}
</style></head><body>
<h1>{html.escape(OBLAST)}</h1>
{stale}
<div class="big">{prob*100:.0f}%</div>
<div class="mut">вероятность тревоги в час после последней записи датасета ({data_end}).
Это эксперимент, а не оповещение. Страница обновлена {now_utc}.</div>
<h2>Качество на тесте (последние {int(TEST_SHARE*100)}% времени)</h2>
<table><tr><th>Модель</th><th>ROC-AUC</th><th>PR-AUC</th></tr>{table}</table>
<div class="mut">Доля тревожных часов в тесте: {share:.1%}.</div>
<h2>Важность признаков</h2>
<table>{imp_rows}</table>
</body></html>"""
    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write(page)


if __name__ == "__main__":
    main()
