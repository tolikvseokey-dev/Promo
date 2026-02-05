import os
import re
import calendar
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import telebot


# =========================
# НАСТРОЙКИ
# =========================
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
if not BOT_TOKEN:
    BOT_TOKEN = "PASTE_YOUR_TOKEN_HERE"  # ← временно, лучше через переменную окружения

DATA_DIR = os.getenv("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)

bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML")


# =========================
# ФОРМАТИРОВАНИЕ / УТИЛИТЫ
# =========================
def _safe_num(x) -> float:
    """Аккуратно парсим числа из Excel (пробелы, ₽, запятые)."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)

    s = str(x).strip()
    s = s.replace("\u00a0", " ").replace(" ", "")
    s = s.replace(",", ".")
    s = re.sub(r"[^0-9\.\-]", "", s)
    if s in ("", "-", "."):
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def fmt_money(x: float) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{x:,.0f}".replace(",", " ")


def fmt_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{x * 100:.1f}%".replace(".", ",")


def fmt_num(x: float, dec: int = 2) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{x:.{dec}f}".replace(".", ",")


def pct_change(a: float, b: float) -> float:
    if b is None or (isinstance(b, float) and (np.isnan(b) or np.isinf(b))):
        return np.nan
    if b == 0:
        return np.nan
    return (a - b) / b


def extract_store_code(val: str) -> Optional[str]:
    """
    Унифицируем код лавки из любой строки вида:
    'М13 ...', 'м 13', 'М13', 'М-13' и т.д.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).upper().strip()
    s = s.replace("М-", "М")
    m = re.search(r"(М\s*\d+)", s)
    if not m:
        return None
    return m.group(1).replace(" ", "")


def iso_prev_week(iso_year: int, iso_week: int) -> Tuple[int, int]:
    """Предыдущая ISO-неделя, корректно на границе года."""
    d = datetime.fromisocalendar(iso_year, iso_week, 1)  # понедельник
    d2 = d - timedelta(days=7)
    iso2 = d2.isocalendar()
    return int(iso2.year), int(iso2.week)


# =========================
# РАСПОЗНАВАНИЕ / ПУТИ ФАЙЛОВ
# =========================
def detect_file_kind(filename: str) -> Tuple[str, int]:
    """
    kind: to | checks | avg | basket | plans | roster | unknown
    year: 25/26 если есть в имени, иначе 0
    """
    name = filename.lower().replace("ё", "е")
    year = 0

    # год из имени
    if re.search(r"(^|[\s_])25([\s_.]|$)", name):
        year = 25
    if re.search(r"(^|[\s_])26([\s_.]|$)", name):
        year = 26

    if "ростер" in name:
        return "roster", 0
    if "план" in name:
        return "plans", 0

    if "длин" in name or "наполн" in name:
        return "basket", year
    if "ср" in name and "чек" in name:
        return "avg", year
    if "чек" in name and "ср" not in name:
        return "checks", year
    if "то" in name or "выручк" in name:
        return "to", year

    return "unknown", year


def path_for(kind: str, year: int) -> str:
    if kind in ("roster", "plans"):
        return os.path.join(DATA_DIR, f"{kind}.xlsx")
    if year in (25, 26):
        return os.path.join(DATA_DIR, f"{kind}_{year}.xlsx")
    return os.path.join(DATA_DIR, f"{kind}.xlsx")


# =========================
# ЧТЕНИЕ ДАННЫХ
# =========================
def read_metric_file(path: str, metric: str) -> pd.DataFrame:
    """
    Ожидаем 4 колонки:
    Региональный управляющий (регион), Торговые точки, <метрика>, ГМД — Дата
    Чтобы не зависеть от названий, берём первые 4 колонки.
    """
    df = pd.read_excel(path)
    df = df.iloc[:, :4].copy()
    df.columns = ["rm_raw", "store_raw", "value", "date"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df["value"] = df["value"].apply(_safe_num)
    df["store_code"] = df["store_raw"].apply(extract_store_code)

    df = df.dropna(subset=["date", "store_code"])
    df["metric"] = metric

    return df[["date", "store_code", "rm_raw", "metric", "value"]]


def load_roster_map(roster_path: str) -> Dict[str, str]:
    """
    Ростер БК.xlsx → лист 'Лавки'
    - столбец '№' = номер лавки (N)
    - столбец 'Регион' = имя РМ
    Маппим: 'М' + № -> РМ
    """
    roster = pd.read_excel(roster_path, sheet_name="Лавки")

    if "№" not in roster.columns:
        raise ValueError("В ростере не нашёл колонку '№'.")
    if "Регион" not in roster.columns:
        raise ValueError("В ростере не нашёл колонку 'Регион'.")

    def _mk_code(x):
        if pd.isna(x):
            return None
        try:
            return f"М{int(x)}"
        except Exception:
            # если вдруг там уже "М13"
            c = extract_store_code(str(x))
            return c

    roster["store_code"] = roster["№"].apply(_mk_code)
    roster["store_code"] = roster["store_code"].astype(str).str.upper().str.strip()

    store_rm = dict(zip(roster["store_code"], roster["Регион"].astype(str).str.strip()))
    return store_rm


def attach_rm(df: pd.DataFrame, store_rm: Dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    out["rm"] = out["store_code"].map(store_rm)
    out["rm"] = out["rm"].fillna(out["rm_raw"])
    return out


def make_wide(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    iso = df["date"].dt.isocalendar()
    df["iso_year"] = iso["year"].astype(int)
    df["iso_week"] = iso["week"].astype(int)

    wide = (
        df.pivot_table(
            index=["date", "iso_year", "iso_week", "store_code", "rm"],
            columns="metric",
            values="value",
            aggfunc="sum",
        )
        .reset_index()
    )

    for col in ["TO", "CHECKS", "BASKET"]:
        if col not in wide.columns:
            wide[col] = np.nan

    # Ср чек — всегда считаем взвешенно
    wide["AVG"] = wide["TO"] / wide["CHECKS"]
    return wide


def read_plans(plans_path: str, store_rm: Dict[str, str]) -> pd.DataFrame:
    """
    Планы: ищем колонки 'Торговые точки' и 'План'.
    """
    df = pd.read_excel(plans_path)

    if "Торговые точки" not in df.columns or "План" not in df.columns:
        # иногда шапка на 2-й строке
        df2 = pd.read_excel(plans_path, header=1)
        if "Торговые точки" in df2.columns and "План" in df2.columns:
            df = df2

    if "Торговые точки" not in df.columns or "План" not in df.columns:
        raise ValueError("В файле планов не нашёл колонки 'Торговые точки' и 'План'.")

    out = df[["Торговые точки", "План"]].copy()
    out = out.rename(columns={"Торговые точки": "store_raw", "План": "plan"})
    out["store_code"] = out["store_raw"].apply(extract_store_code)
    out["plan"] = out["plan"].apply(_safe_num)
    out = out.dropna(subset=["store_code"])
    out["rm"] = out["store_code"].map(store_rm)

    return out[["store_code", "rm", "plan"]]


# =========================
# РАСЧЁТЫ
# =========================
def period_mtd(last_date: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    return last_date.replace(day=1), last_date


def period_mtd_prev_year(last_date: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    # аналогичный период прошлого календарного года
    y = int(last_date.year) - 1
    return pd.Timestamp(year=y, month=int(last_date.month), day=1), pd.Timestamp(year=y, month=int(last_date.month), day=int(last_date.day))


def plan_to_date(month_plan: float, last_date: pd.Timestamp) -> float:
    """
    План на дату = месячный план * (текущий день месяца / число дней в месяце)
    Это простая и понятная логика "плана на дату".
    """
    if month_plan is None or (isinstance(month_plan, float) and np.isnan(month_plan)):
        return np.nan
    days_in_month = calendar.monthrange(int(last_date.year), int(last_date.month))[1]
    ratio = int(last_date.day) / days_in_month
    return float(month_plan) * ratio


def network_mtd(w: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, float]:
    d = w[(w["date"] >= start) & (w["date"] <= end)]
    to = float(np.nansum(d["TO"]))
    checks = float(np.nansum(d["CHECKS"]))
    avg = to / checks if checks else np.nan
    basket = float(np.nansum(d["BASKET"] * d["CHECKS"]) / checks) if checks else np.nan
    return {"to": to, "checks": checks, "avg": avg, "basket": basket}


def per_store_period(w: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    d = w[(w["date"] >= start) & (w["date"] <= end)]
    g = d.groupby("store_code").agg(TO=("TO", "sum"), CHECKS=("CHECKS", "sum")).reset_index()
    g["AVG"] = g["TO"] / g["CHECKS"]
    return g


def network_week(w: pd.DataFrame, iso_year: int, iso_week: int) -> Dict[str, float]:
    d = w[(w["iso_year"] == iso_year) & (w["iso_week"] == iso_week)]
    to = float(np.nansum(d["TO"]))
    checks = float(np.nansum(d["CHECKS"]))
    avg = to / checks if checks else np.nan
    return {"to": to, "checks": checks, "avg": avg}


def top_anti_3(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    top = s.sort_values(ascending=False).head(3)
    anti = s.sort_values(ascending=True).head(3)
    return top, anti


# =========================
# СБОРКА ОТЧЁТА (как ты утвердил)
# =========================
def build_report() -> str:
    required = [
        path_for("roster", 0),
        path_for("plans", 0),
        path_for("to", 25),
        path_for("checks", 25),
        path_for("to", 26),
        path_for("checks", 26),
        path_for("basket", 26),
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        return "❌ Не хватает файлов:\n" + "\n".join([f"• {os.path.basename(x)}" for x in missing])

    store_rm = load_roster_map(path_for("roster", 0))

    # 26
    df26 = pd.concat(
        [
            read_metric_file(path_for("to", 26), "TO"),
            read_metric_file(path_for("checks", 26), "CHECKS"),
            read_metric_file(path_for("basket", 26), "BASKET"),
        ],
        ignore_index=True,
    )
    df26 = attach_rm(df26, store_rm)
    w26 = make_wide(df26)

    # 25 (для LFL)
    df25 = pd.concat(
        [
            read_metric_file(path_for("to", 25), "TO"),
            read_metric_file(path_for("checks", 25), "CHECKS"),
        ],
        ignore_index=True,
    )
    df25 = attach_rm(df25, store_rm)
    w25 = make_wide(df25)

    # last date = последняя дата в 26 (как “на дату отчёта”)
    last_date = pd.to_datetime(w26["date"].max())
    mtd_start_26, mtd_end_26 = period_mtd(last_date)
    mtd_start_25, mtd_end_25 = period_mtd_prev_year(last_date)

    # MTD сеть 26
    net26 = network_mtd(w26, mtd_start_26, mtd_end_26)

    # планы (месячный план) + план на дату
    plans = read_plans(path_for("plans", 0), store_rm)

    stores_in_mtd = set(w26[(w26["date"] >= mtd_start_26) & (w26["date"] <= mtd_end_26)]["store_code"].unique())
    plans_used = plans[plans["store_code"].isin(stores_in_mtd)].copy()

    month_plan_total = float(np.nansum(plans_used["plan"]))
    plan_on_date_total = plan_to_date(month_plan_total, last_date)
    perf_net = net26["to"] / plan_on_date_total if plan_on_date_total and not np.isnan(plan_on_date_total) else np.nan

    # РМ — показать ВСЕХ (по тем, кто есть в ростере/планах/факте)
    fact_by_rm = (
        w26[(w26["date"] >= mtd_start_26) & (w26["date"] <= mtd_end_26)]
        .groupby("rm", as_index=False)["TO"]
        .sum()
        .rename(columns={"TO": "fact"})
    )
    plans_by_rm = plans_used.groupby("rm", as_index=False)["plan"].sum().rename(columns={"plan": "month_plan"})
    rm = fact_by_rm.merge(plans_by_rm, on="rm", how="outer")

    # план на дату по РМ
    rm["plan_on_date"] = rm["month_plan"].apply(lambda x: plan_to_date(x, last_date))
    rm["perf"] = rm["fact"] / rm["plan_on_date"]
    rm = rm.sort_values("perf", ascending=False)

    # LFL (MTD) сеть — только пересечение лавок (25 и 26)
    s26 = per_store_period(w26, mtd_start_26, mtd_end_26).set_index("store_code")
    s25 = per_store_period(w25, mtd_start_25, mtd_end_25).set_index("store_code")
    common = sorted(set(s26.index).intersection(set(s25.index)))

    # сетевой LFL на пересечении
    to26_lfl = float(np.nansum(s26.loc[common, "TO"])) if common else np.nan
    to25_lfl = float(np.nansum(s25.loc[common, "TO"])) if common else np.nan
    ch26_lfl = float(np.nansum(s26.loc[common, "CHECKS"])) if common else np.nan
    ch25_lfl = float(np.nansum(s25.loc[common, "CHECKS"])) if common else np.nan

    avg26_lfl = to26_lfl / ch26_lfl if ch26_lfl else np.nan
    avg25_lfl = to25_lfl / ch25_lfl if ch25_lfl else np.nan

    lfl_to = pct_change(to26_lfl, to25_lfl)
    lfl_checks = pct_change(ch26_lfl, ch25_lfl)
    lfl_avg = pct_change(avg26_lfl, avg25_lfl)

    # ТОП/АНТИ-3 LFL (по лавкам) для ТО / ЧЕКИ / СР.ЧЕК
    yoy = pd.DataFrame(index=common)
    if common:
        yoy["TO"] = (s26.loc[common, "TO"] - s25.loc[common, "TO"]) / s25.loc[common, "TO"]
        yoy["CHECKS"] = (s26.loc[common, "CHECKS"] - s25.loc[common, "CHECKS"]) / s25.loc[common, "CHECKS"]
        yoy["AVG"] = ((s26.loc[common, "TO"] / s26.loc[common, "CHECKS"]) - (s25.loc[common, "TO"] / s25.loc[common, "CHECKS"])) / (s25.loc[common, "TO"] / s25.loc[common, "CHECKS"])
        yoy = yoy.replace([np.inf, -np.inf], np.nan)

    top_to, anti_to = top_anti_3(yoy["TO"]) if common else (pd.Series(dtype=float), pd.Series(dtype=float))
    top_checks, anti_checks = top_anti_3(yoy["CHECKS"]) if common else (pd.Series(dtype=float), pd.Series(dtype=float))
    top_avg, anti_avg = top_anti_3(yoy["AVG"]) if common else (pd.Series(dtype=float), pd.Series(dtype=float))

    # Неделя к неделе (по номеру недели)
    iso = last_date.to_pydatetime().isocalendar()
    iso_y = int(iso.year)
    iso_w = int(iso.week)
    prev_y, prev_w = iso_prev_week(iso_y, iso_w)

    wk26 = network_week(w26, iso_y, iso_w)
    wk26_prev = network_week(w26, prev_y, prev_w)

    w26_to = pct_change(wk26["to"], wk26_prev["to"])
    w26_checks = pct_change(wk26["checks"], wk26_prev["checks"])
    w26_avg = pct_change(wk26["avg"], wk26_prev["avg"])

    # та же пара недель, но год -1 (ISO)
    wk25 = network_week(w25, iso_y - 1, iso_w)
    wk25_prev = network_week(w25, prev_y - 1, prev_w)

    w25_to = pct_change(wk25["to"], wk25_prev["to"])
    w25_checks = pct_change(wk25["checks"], wk25_prev["checks"])
    w25_avg = pct_change(wk25["avg"], wk25_prev["avg"])

    # ====== Формируем сообщение (одним блоком) ======
    period_str = f"{mtd_start_26:%d.%m}–{mtd_end_26:%d.%m}"

    lines: List[str] = []

    lines.append(f"📊 <b>АНАЛИТИКА СЕТИ</b> | MTD ({period_str})")
    lines.append("")
    lines.append(f"ТО Факт: <b>{fmt_money(net26['to'])} ₽</b>")
    lines.append(f"ТО План на дату: <b>{fmt_money(plan_on_date_total)} ₽</b>")
    lines.append(f"Выполнение плана: <b>{fmt_pct(perf_net)}</b>")
    lines.append("")
    lines.append(f"Чеки: <b>{fmt_money(net26['checks'])}</b>")
    lines.append(f"Ср. чек: <b>{fmt_money(net26['avg'])} ₽</b>")
    lines.append(f"Длина чека: <b>{fmt_num(net26['basket'], 2)}</b>")
    lines.append("")
    lines.append("👥 <b>РМ</b> | выполнение плана (MTD)")

    # ВСЕ РМ
    for _, r in rm.iterrows():
        rm_name = str(r["rm"]) if pd.notna(r["rm"]) else "—"
        lines.append(f"{rm_name} — <b>{fmt_pct(r['perf'])}</b>")

    lines.append("")
    lines.append("📈 <b>LFL</b> | MTD (2026 vs 2025)")
    lines.append(f"ТО: <b>{fmt_pct(lfl_to)}</b>")
    lines.append(f"Чеки: <b>{fmt_pct(lfl_checks)}</b>")
    lines.append(f"Ср. чек: <b>{fmt_pct(lfl_avg)}</b>")

    def render_top_block(title: str, s: pd.Series):
        lines.append("")
        lines.append(title)
        if s is None or len(s) == 0:
            lines.append("—")
            return
        for i, (k, v) in enumerate(s.items(), start=1):
            lines.append(f"{i}) {k}  <b>{fmt_pct(v)}</b>")

    render_top_block("🔥 <b>ТОП-3 LFL (MTD) — ТО</b>", top_to)
    render_top_block("❄️ <b>АНТИ-ТОП-3 LFL (MTD) — ТО</b>", anti_to)

    render_top_block("🔥 <b>ТОП-3 LFL (MTD) — Чеки</b>", top_checks)
    render_top_block("❄️ <b>АНТИ-ТОП-3 LFL (MTD) — Чеки</b>", anti_checks)

    render_top_block("🔥 <b>ТОП-3 LFL (MTD) — Ср. чек</b>", top_avg)
    render_top_block("❄️ <b>АНТИ-ТОП-3 LFL (MTD) — Ср. чек</b>", anti_avg)

    lines.append("")
    lines.append(f"📊 <b>НЕДЕЛЯ К НЕДЕЛЕ</b> | LFL 2026 (неделя {iso_w} vs {prev_w})")
    lines.append(f"ТО: <b>{fmt_pct(w26_to)}</b>")
    lines.append(f"Чеки: <b>{fmt_pct(w26_checks)}</b>")
    lines.append(f"Ср. чек: <b>{fmt_pct(w26_avg)}</b>")

    lines.append("")
    lines.append(f"📊 <b>НЕДЕЛЯ К НЕДЕЛЕ</b> | LFL 2025 (неделя {iso_w} vs {prev_w})")
    lines.append(f"ТО: <b>{fmt_pct(w25_to)}</b>")
    lines.append(f"Чеки: <b>{fmt_pct(w25_checks)}</b>")
    lines.append(f"Ср. чек: <b>{fmt_pct(w25_avg)}</b>")

    # Выводы — пока шаблонные (без “автогенерации”), но уже в правильных терминах
    lines.append("")
    lines.append("🧠 <b>ВЫВОДЫ</b>")
    lines.append(f"1) LFL MTD: ТО {fmt_pct(lfl_to)}, Чеки {fmt_pct(lfl_checks)}, Ср. чек {fmt_pct(lfl_avg)} — смотрим баланс трафика и среднего чека.")
    lines.append(f"2) Выполнение плана по сети: {fmt_pct(perf_net)} — при текущем темпе возможен риск недобора.")
    lines.append("3) Фокус недели — лавки АНТИ-ТОП-3 по LFL: они дают непропорционально большой минус сети.")
    lines.append("4) Сильные точки из ТОП-3 удерживают динамику — важно масштабировать их практики.")
    lines.append("5) Если LFL по чекам уходит в минус — усиливаем трафик (витрина/промо/ассортимент/контроль наличия).")

    return "\n".join(lines)


# =========================
# TELEGRAM HANDLERS
# =========================
@bot.message_handler(commands=["start"])
def cmd_start(m):
    bot.send_message(
        m.chat.id,
        "Привет! 👋\n"
        "Загрузи Excel-файлы (как документы), потом вызови /report.\n\n"
        "Команды:\n"
        "• /files — что загружено\n"
        "• /report — аналитика одним сообщением"
    )


@bot.message_handler(commands=["files"])
def cmd_files(m):
    names = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".xlsx")])
    if not names:
        bot.send_message(m.chat.id, "Файлы не загружены 🙂 Пришли Excel документами.")
        return
    bot.send_message(m.chat.id, "Загружено:\n" + "\n".join([f"• {x}" for x in names]))


@bot.message_handler(commands=["report"])
def cmd_report(m):
    try:
        text = build_report()
    except Exception as e:
        text = f"❌ Ошибка при расчёте: {e}"
    bot.send_message(m.chat.id, text)


@bot.message_handler(content_types=["document"])
def on_document(m):
    doc = m.document
    kind, year = detect_file_kind(doc.file_name)

    if kind == "unknown":
        bot.send_message(
            m.chat.id,
            "Не понял тип файла 🤔\n"
            "Назови файл так, чтобы было видно что это:\n"
            "• ТО 25 / ТО 26\n"
            "• чеки 25 / чеки 26\n"
            "• длина 26\n"
            "• планы\n"
            "• ростер\n\n"
            "Файлы 'ср чек' можно не грузить — ср. чек считаю как ТО/Чеки."
        )
        return

    save_path = path_for(kind, year)
    file_info = bot.get_file(doc.file_id)
    downloaded = bot.download_file(file_info.file_path)

    with open(save_path, "wb") as f:
        f.write(downloaded)

    bot.send_message(
        m.chat.id,
        f"✅ Сохранил: <b>{os.path.basename(save_path)}</b>\n"
        f"Тип: <b>{kind.upper()}</b>  Год: <b>{year if year else '—'}</b>"
    )


if __name__ == "__main__":
    print("Bot is running...")
    bot.infinity_polling(timeout=60, long_polling_timeout=60)
