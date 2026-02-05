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
    BOT_TOKEN = "PASTE_YOUR_TOKEN_HERE"  # лучше через переменную окружения

DATA_DIR = os.getenv("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)

bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML")

# состояния ввода даты для /report
WAITING_FOR_REPORT_DATE: Dict[int, bool] = {}


# =========================
# ФОРМАТИРОВАНИЕ / УТИЛИТЫ
# =========================
def _safe_num(x) -> float:
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
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).upper().strip()
    s = s.replace("М-", "М")
    m = re.search(r"(М\s*\d+)", s)
    if not m:
        return None
    return m.group(1).replace(" ", "")


def _norm_header(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).replace("\u00a0", " ")
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def parse_input_date(text: str) -> Optional[datetime]:
    """
    Ожидаем DD.MM.YY (например 27.01.26).
    Интерпретируем YY как 2000+YY.
    """
    t = text.strip()
    m = re.fullmatch(r"(\d{2})\.(\d{2})\.(\d{2})", t)
    if not m:
        return None
    dd, mm, yy = map(int, m.groups())
    yyyy = 2000 + yy
    try:
        return datetime(yyyy, mm, dd)
    except ValueError:
        return None


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
    Метрики: НЕ читаем по названиям столбцов.
    Берём первые 4 столбца в порядке:
    1) РМ, 2) Торговые точки, 3) значение, 4) дата
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
            return extract_store_code(str(x))

    roster["store_code"] = roster["№"].apply(_mk_code)
    roster["store_code"] = roster["store_code"].astype(str).str.upper().str.strip()

    return dict(zip(roster["store_code"], roster["Регион"].astype(str).str.strip()))


def attach_rm(df: pd.DataFrame, store_rm: Dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    out["rm"] = out["store_code"].map(store_rm)
    out["rm"] = out["rm"].fillna(out["rm_raw"])
    return out


def make_wide(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    wide = (
        df.pivot_table(
            index=["date", "store_code", "rm"],
            columns="metric",
            values="value",
            aggfunc="sum",
        )
        .reset_index()
    )
    for col in ["TO", "CHECKS", "BASKET"]:
        if col not in wide.columns:
            wide[col] = np.nan
    # Ср. чек — взвешенно
    wide["AVG"] = wide["TO"] / wide["CHECKS"]
    return wide


def read_plans(plans_path: str, store_rm: Dict[str, str]) -> pd.DataFrame:
    """
    План — читаем по заголовкам, но шапка может быть не в первой строке.
    Ищем строку, где одновременно есть 'торговые точки' и 'план', и используем её как header.
    """
    raw = pd.read_excel(plans_path, header=None)
    header_row = None
    for i in range(min(50, len(raw))):
        row_vals = [_norm_header(v) for v in raw.iloc[i].tolist()]
        if ("торговые точки" in row_vals) and ("план" in row_vals):
            header_row = i
            break

    if header_row is None:
        for guess in range(0, 15):
            df_try = pd.read_excel(plans_path, header=guess)
            cols_norm = [_norm_header(c) for c in df_try.columns]
            if ("торговые точки" in cols_norm) and ("план" in cols_norm):
                header_row = guess
                break

    if header_row is None:
        raise ValueError("В файле планов не нашёл строку заголовков с 'Торговые точки' и 'План'.")

    df = pd.read_excel(plans_path, header=header_row)
    col_map = {_norm_header(c): c for c in df.columns}

    if "торговые точки" not in col_map or "план" not in col_map:
        raise ValueError("В файле планов не нашёл колонки 'Торговые точки' и 'План' (после распознавания шапки).")

    store_col = col_map["торговые точки"]
    plan_col = col_map["план"]

    out = df[[store_col, plan_col]].copy()
    out = out.rename(columns={store_col: "store_raw", plan_col: "month_plan"})
    out["store_code"] = out["store_raw"].apply(extract_store_code)
    out["month_plan"] = out["month_plan"].apply(_safe_num)
    out = out.dropna(subset=["store_code"])
    out["rm"] = out["store_code"].map(store_rm)

    return out[["store_code", "rm", "month_plan"]]


# =========================
# ПЕРИОДЫ (по введённой дате)
# =========================
def period_mtd(report_date: datetime) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(report_date.year, report_date.month, 1)
    end = pd.Timestamp(report_date.year, report_date.month, report_date.day)
    return start, end


def period_last_week_25(report_date: datetime) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    "Последняя неделя месяца" по твоей логике:
    25-е число -> введённая дата (включительно)
    """
    if report_date.day < 25:
        raise ValueError("Для блока 'последняя неделя' введи дату с 25 по конец месяца.")
    start = pd.Timestamp(report_date.year, report_date.month, 25)
    end = pd.Timestamp(report_date.year, report_date.month, report_date.day)
    return start, end


def same_period_prev_year(start: pd.Timestamp, end: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    return (
        pd.Timestamp(start.year - 1, start.month, start.day),
        pd.Timestamp(end.year - 1, end.month, end.day),
    )


def plan_to_date(month_plan: float, report_date: datetime) -> float:
    if month_plan is None or (isinstance(month_plan, float) and np.isnan(month_plan)):
        return np.nan
    days_in_month = calendar.monthrange(report_date.year, report_date.month)[1]
    ratio = report_date.day / days_in_month
    return float(month_plan) * ratio


# =========================
# АГРЕГАЦИИ
# =========================
def network_metrics(w: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, float]:
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


def top_anti_3(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    return s.sort_values(ascending=False).head(3), s.sort_values(ascending=True).head(3)


# =========================
# СБОРКА ОТЧЁТА
# =========================
def build_report(report_date: datetime) -> str:
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

    # 2026
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

    # 2025
    df25 = pd.concat(
        [
            read_metric_file(path_for("to", 25), "TO"),
            read_metric_file(path_for("checks", 25), "CHECKS"),
        ],
        ignore_index=True,
    )
    df25 = attach_rm(df25, store_rm)
    w25 = make_wide(df25)

    # Проверим, что введённая дата есть в данных 2026 (хотя бы где-то)
    report_ts = pd.Timestamp(report_date.year, report_date.month, report_date.day)
    if w26[(w26["date"] == report_ts)].empty:
        return (
            "❌ В данных 2026 нет записей за введённую дату.\n"
            f"Ты ввёл: {report_date:%d.%m.%y}\n"
            "Проверь, что файлы ТО/чеки/длина 26 загружены и содержат эту дату."
        )

    # Периоды
    mtd_start_26, mtd_end_26 = period_mtd(report_date)
    mtd_start_25, mtd_end_25 = same_period_prev_year(mtd_start_26, mtd_end_26)

    # "последняя неделя" (25 -> дата)
    lastw_start_26, lastw_end_26 = period_last_week_25(report_date)
    lastw_start_25, lastw_end_25 = same_period_prev_year(lastw_start_26, lastw_end_26)

    # MTD сеть 2026
    net_mtd_26 = network_metrics(w26, mtd_start_26, mtd_end_26)

    # Планы
    plans = read_plans(path_for("plans", 0), store_rm)

    # планы считаем по лавкам, которые реально есть в периоде MTD
    stores_in_mtd = set(w26[(w26["date"] >= mtd_start_26) & (w26["date"] <= mtd_end_26)]["store_code"].unique())
    plans_used = plans[plans["store_code"].isin(stores_in_mtd)].copy()

    month_plan_total = float(np.nansum(plans_used["month_plan"]))
    plan_on_date_total = plan_to_date(month_plan_total, report_date)
    perf_net = net_mtd_26["to"] / plan_on_date_total if plan_on_date_total and not np.isnan(plan_on_date_total) else np.nan

    # РМ — полный список (по факту/плану)
    fact_by_rm = (
        w26[(w26["date"] >= mtd_start_26) & (w26["date"] <= mtd_end_26)]
        .groupby("rm", as_index=False)["TO"]
        .sum()
        .rename(columns={"TO": "fact"})
    )
    plans_by_rm = plans_used.groupby("rm", as_index=False)["month_plan"].sum()
    rm = fact_by_rm.merge(plans_by_rm, on="rm", how="outer")
    rm["plan_on_date"] = rm["month_plan"].apply(lambda x: plan_to_date(x, report_date))
    rm["perf"] = rm["fact"] / rm["plan_on_date"]
    rm = rm.sort_values("perf", ascending=False)

    # LFL MTD (пересечение лавок)
    s26_mtd = per_store_period(w26, mtd_start_26, mtd_end_26).set_index("store_code")
    s25_mtd = per_store_period(w25, mtd_start_25, mtd_end_25).set_index("store_code")
    common_mtd = sorted(set(s26_mtd.index).intersection(set(s25_mtd.index)))

    to26_lfl = float(np.nansum(s26_mtd.loc[common_mtd, "TO"])) if common_mtd else np.nan
    to25_lfl = float(np.nansum(s25_mtd.loc[common_mtd, "TO"])) if common_mtd else np.nan
    ch26_lfl = float(np.nansum(s26_mtd.loc[common_mtd, "CHECKS"])) if common_mtd else np.nan
    ch25_lfl = float(np.nansum(s25_mtd.loc[common_mtd, "CHECKS"])) if common_mtd else np.nan

    avg26_lfl = to26_lfl / ch26_lfl if ch26_lfl else np.nan
    avg25_lfl = to25_lfl / ch25_lfl if ch25_lfl else np.nan

    lfl_to = pct_change(to26_lfl, to25_lfl)
    lfl_checks = pct_change(ch26_lfl, ch25_lfl)
    lfl_avg = pct_change(avg26_lfl, avg25_lfl)

    # ТОП/АНТИ-3 LFL (MTD) — по лавкам
    yoy = pd.DataFrame(index=common_mtd)
    if common_mtd:
        yoy["TO"] = (s26_mtd.loc[common_mtd, "TO"] - s25_mtd.loc[common_mtd, "TO"]) / s25_mtd.loc[common_mtd, "TO"]
        yoy["CHECKS"] = (s26_mtd.loc[common_mtd, "CHECKS"] - s25_mtd.loc[common_mtd, "CHECKS"]) / s25_mtd.loc[common_mtd, "CHECKS"]
        yoy["AVG"] = (
            (s26_mtd.loc[common_mtd, "TO"] / s26_mtd.loc[common_mtd, "CHECKS"])
            - (s25_mtd.loc[common_mtd, "TO"] / s25_mtd.loc[common_mtd, "CHECKS"])
        ) / (s25_mtd.loc[common_mtd, "TO"] / s25_mtd.loc[common_mtd, "CHECKS"])
        yoy = yoy.replace([np.inf, -np.inf], np.nan)

    top_to, anti_to = top_anti_3(yoy["TO"]) if common_mtd else (pd.Series(dtype=float), pd.Series(dtype=float))
    top_checks, anti_checks = top_anti_3(yoy["CHECKS"]) if common_mtd else (pd.Series(dtype=float), pd.Series(dtype=float))
    top_avg, anti_avg = top_anti_3(yoy["AVG"]) if common_mtd else (pd.Series(dtype=float), pd.Series(dtype=float))

    # Блок "последняя неделя (25->дата)" — и её LFL сравнение (с тем же периодом прошлого года)
    net_lastw_26 = network_metrics(w26, lastw_start_26, lastw_end_26)
    net_lastw_25 = network_metrics(w25, lastw_start_25, lastw_end_25)

    lfl_lastw_to = pct_change(net_lastw_26["to"], net_lastw_25["to"])
    lfl_lastw_checks = pct_change(net_lastw_26["checks"], net_lastw_25["checks"])
    lfl_lastw_avg = pct_change(net_lastw_26["avg"], net_lastw_25["avg"])

    # ========= Формируем сообщение =========
    period_mtd_str = f"{mtd_start_26:%d.%m}–{mtd_end_26:%d.%m}"
    period_lastw_str = f"{lastw_start_26:%d.%m}–{lastw_end_26:%d.%m}"

    lines: List[str] = []

    lines.append(f"📊 <b>АНАЛИТИКА СЕТИ</b> | MTD ({period_mtd_str})")
    lines.append(f"Дата отчёта: <b>{report_date:%d.%m.%y}</b>")
    lines.append("")
    lines.append(f"ТО Факт: <b>{fmt_money(net_mtd_26['to'])} ₽</b>")
    lines.append(f"ТО План на дату: <b>{fmt_money(plan_on_date_total)} ₽</b>")
    lines.append(f"Выполнение плана: <b>{fmt_pct(perf_net)}</b>")
    lines.append("")
    lines.append(f"Чеки: <b>{fmt_money(net_mtd_26['checks'])}</b>")
    lines.append(f"Ср. чек: <b>{fmt_money(net_mtd_26['avg'])} ₽</b>")
    lines.append(f"Длина чека: <b>{fmt_num(net_mtd_26['basket'], 2)}</b>")

    lines.append("")
    lines.append("👥 <b>РМ</b> | выполнение плана (MTD)")
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
    lines.append(f"📊 <b>ПОСЛЕДНЯЯ НЕДЕЛЯ МЕСЯЦА</b> (по правилу 25→дата) | {period_lastw_str}")
    lines.append("📌 LFL (2026 vs 2025) по этому же периоду")
    lines.append(f"ТО: <b>{fmt_pct(lfl_lastw_to)}</b>")
    lines.append(f"Чеки: <b>{fmt_pct(lfl_lastw_checks)}</b>")
    lines.append(f"Ср. чек: <b>{fmt_pct(lfl_lastw_avg)}</b>")

    lines.append("")
    lines.append("🧠 <b>ВЫВОДЫ</b>")
    lines.append(
        f"1) LFL MTD: ТО {fmt_pct(lfl_to)}, Чеки {fmt_pct(lfl_checks)}, Ср. чек {fmt_pct(lfl_avg)} — баланс трафика и среднего чека."
    )
    lines.append(
        f"2) Выполнение плана по сети: {fmt_pct(perf_net)} (план на дату) — при текущем темпе возможен риск недобора."
    )
    lines.append(
        "3) Фокус — лавки АНТИ-ТОП-3 по LFL: они дают непропорционально большой минус сети."
    )
    lines.append(
        f"4) По последней неделе (25→дата) LFL: ТО {fmt_pct(lfl_lastw_to)}, Чеки {fmt_pct(lfl_lastw_checks)}, Ср. чек {fmt_pct(lfl_lastw_avg)} — быстрый индикатор конца месяца."
    )

    return "\n".join(lines)


# =========================
# TELEGRAM HANDLERS
# =========================
@bot.message_handler(commands=["start"])
def cmd_start(m):
    WAITING_FOR_REPORT_DATE[m.chat.id] = False
    bot.send_message(
        m.chat.id,
        "Привет! 👋\n"
        "Загрузи Excel-файлы (как документы), потом вызови /report.\n\n"
        "Команды:\n"
        "• /files — что загружено\n"
        "• /report — запросить дату и сформировать отчёт"
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
    WAITING_FOR_REPORT_DATE[m.chat.id] = True
    bot.send_message(
        m.chat.id,
        "Введи дату для анализа в формате <b>DD.MM.YY</b>\n"
        "Пример: <b>27.01.26</b>"
    )


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


@bot.message_handler(func=lambda msg: True, content_types=["text"])
def on_text(m):
    # если ждём дату — обрабатываем как дату отчёта
    if WAITING_FOR_REPORT_DATE.get(m.chat.id, False):
        dt = parse_input_date(m.text)
        if not dt:
            bot.send_message(
                m.chat.id,
                "❌ Неверный формат даты.\n"
                "Введи дату в формате <b>DD.MM.YY</b>, например <b>27.01.26</b>."
            )
            return

        # пробуем построить отчёт
        try:
            text = build_report(dt)
        except Exception as e:
            text = f"❌ Ошибка при расчёте: {e}"

        WAITING_FOR_REPORT_DATE[m.chat.id] = False
        bot.send_message(m.chat.id, text)
        return

    # обычный текст вне режима даты
    # (можно оставить подсказку, чтобы не молчал)
    if m.text.strip().startswith("/"):
        return
    bot.send_message(m.chat.id, "Напиши /report чтобы сделать отчёт 🙂")


if __name__ == "__main__":
    print("Bot is running...")
    bot.infinity_polling(timeout=60, long_polling_timeout=60)
