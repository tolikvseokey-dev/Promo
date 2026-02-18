import os
import re
import calendar
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import telebot
from telebot import types


# =========================
# НАСТРОЙКИ
# =========================
BOT_VERSION = "analytics-bot-2026-02-17-modeA-top5-rm-no-tops-cut50"

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
if not BOT_TOKEN:
    BOT_TOKEN = "PASTE_YOUR_TOKEN_HERE"  # лучше через переменную окружения

DATA_DIR = os.getenv("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)

bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML")

# Состояния
WAITING_FOR_REPORT_DATE: Dict[int, bool] = {}     # ждём дату
REPORT_MODE: Dict[int, str] = {}                  # "network" | "rm"
SELECTED_RM: Dict[int, Optional[str]] = {}        # выбранный РМ для режима rm
RM_OPTIONS: Dict[int, List[str]] = {}             # список РМ для кнопок выбора (по chat_id)


# =========================
# УТИЛИТЫ / ФОРМАТЫ
# =========================
SEP = "━━━━━━━━━━━━━━━━━━━━━━"

def send_long(chat_id: int, text: str):
    """Отправка длинных сообщений частями (лимит Telegram ~4096 символов)."""
    if not text:
        return
    limit = 3900
    t = text
    while len(t) > limit:
        cut = t.rfind("\n", 0, limit)
        if cut < 500:
            cut = limit
        bot.send_message(chat_id, t[:cut])
        t = t[cut:].lstrip("\n")
    if t:
        bot.send_message(chat_id, t)


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


def fmt_pct_signed(x: float) -> str:
    """Для LFL / динамик: всегда со знаком +/− (если число)."""
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    sign = "+" if x > 0 else ""
    return (sign + f"{x * 100:.1f}%").replace(".", ",")


def fmt_pct_plain(x: float) -> str:
    """Для выполнения плана: без плюса."""
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
    Принимаем разные форматы даты, например:
    - 310126
    - 31.01.26
    - 31\\01\\26
    - 31/01/26
    - 31ю01ю26
    - 31,01,26
    - 31-01-26
    - 31 01 26
    - 31.1.26
    - 31.01.2026
    - 31012026
    """
    if text is None:
        return None

    t = str(text).strip()

    t = t.replace("ю", ".").replace("Ю", ".")
    t = t.replace("\\", ".").replace("/", ".").replace(",", ".").replace("-", ".").replace("_", ".")
    t = re.sub(r"\s+", " ", t).strip()

    digits_only = re.sub(r"\D", "", t)

    def _make(dd: int, mm: int, yyyy: int) -> Optional[datetime]:
        try:
            return datetime(yyyy, mm, dd)
        except ValueError:
            return None

    if len(digits_only) == 6:  # ddmmyy
        dd = int(digits_only[0:2])
        mm = int(digits_only[2:4])
        yy = int(digits_only[4:6])
        return _make(dd, mm, 2000 + yy)

    if len(digits_only) == 8:  # ddmmyyyy
        dd = int(digits_only[0:2])
        mm = int(digits_only[2:4])
        yyyy = int(digits_only[4:8])
        if 1900 <= yyyy <= 2100:
            return _make(dd, mm, yyyy)
        return None

    parts = re.findall(r"\d+", t)
    if len(parts) >= 3:
        dd = int(parts[0])
        mm = int(parts[1])
        yy_or_yyyy = parts[2]

        if len(yy_or_yyyy) == 2:
            yyyy = 2000 + int(yy_or_yyyy)
        elif len(yy_or_yyyy) == 4:
            yyyy = int(yy_or_yyyy)
        else:
            return None

        if not (1900 <= yyyy <= 2100):
            return None
        return _make(dd, mm, yyyy)

    return None


def iso_week_year(d: datetime) -> Tuple[int, int]:
    iso = d.isocalendar()
    return int(iso.year), int(iso.week)


def prev_week_of(d: datetime) -> Tuple[int, int]:
    d2 = d - timedelta(days=7)
    return iso_week_year(d2)


# =========================
# ФАЙЛЫ / РАСПОЗНАВАНИЕ
# =========================
def detect_file_kind(filename: str) -> Tuple[str, int]:
    """
    kind: to | checks | basket | plans | roster | unknown
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
    Берём первые 4 столбца:
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


def load_roster_maps(roster_path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Возвращает:
    - store_code -> RM (Регион)
    - store_code -> store_name (Лавка)
    """
    roster = pd.read_excel(roster_path, sheet_name="Лавки")

    if "№" not in roster.columns:
        raise ValueError("В ростере не нашла колонку '№'.")
    if "Регион" not in roster.columns:
        raise ValueError("В ростере не нашла колонку 'Регион'.")
    if "Лавка" not in roster.columns:
        raise ValueError("В ростере не нашла колонку 'Лавка'.")

    def _mk_code(x):
        if pd.isna(x):
            return None
        try:
            return f"М{int(x)}"
        except Exception:
            return extract_store_code(str(x))

    roster["store_code"] = roster["№"].apply(_mk_code)
    roster["store_code"] = roster["store_code"].astype(str).str.upper().str.strip()

    store_to_rm = dict(zip(roster["store_code"], roster["Регион"].astype(str).str.strip()))
    store_to_name = dict(zip(roster["store_code"], roster["Лавка"].astype(str).str.strip()))
    return store_to_rm, store_to_name


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

    wide["AVG"] = wide["TO"] / wide["CHECKS"]
    return wide


def read_plans(plans_path: str, store_rm: Dict[str, str]) -> pd.DataFrame:
    """
    Ищем строку, где одновременно есть 'торговые точки' и 'план', и используем её как header.
    """
    raw = pd.read_excel(plans_path, header=None)
    header_row = None
    for i in range(min(60, len(raw))):
        row_vals = [_norm_header(v) for v in raw.iloc[i].tolist()]
        if ("торговые точки" in row_vals) and ("план" in row_vals):
            header_row = i
            break

    if header_row is None:
        for guess in range(0, 20):
            df_try = pd.read_excel(plans_path, header=guess)
            cols_norm = [_norm_header(c) for c in df_try.columns]
            if ("торговые точки" in cols_norm) and ("план" in cols_norm):
                header_row = guess
                break

    if header_row is None:
        raise ValueError("В файле планов не нашла строку заголовков с 'Торговые точки' и 'План'.")

    df = pd.read_excel(plans_path, header=header_row)
    col_map = {_norm_header(c): c for c in df.columns}

    if "торговые точки" not in col_map or "план" not in col_map:
        raise ValueError("В файле планов не нашла колонки 'Торговые точки' и 'План'.")

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
# ПЕРИОДЫ
# =========================
def period_mtd(report_date: datetime) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(report_date.year, report_date.month, 1)
    end = pd.Timestamp(report_date.year, report_date.month, report_date.day)
    return start, end


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


def top_anti_n(series: pd.Series, n: int = 5, min_pct: float = -0.5, max_pct: float = 0.5) -> Tuple[pd.Series, pd.Series]:
    """
    ТОП/АНТИ-ТОП с фильтром выбросов:
    исключаем значения > +50% и < -50% (по умолчанию).
    """
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    s = s[(s >= min_pct) & (s <= max_pct)]
    top = s.sort_values(ascending=False).head(n)
    anti = s.sort_values(ascending=True).head(n)
    return top, anti


def weekly_network(w: pd.DataFrame, iso_year: int, iso_week: int) -> Dict[str, float]:
    d = w[(w["iso_year"] == iso_year) & (w["iso_week"] == iso_week)]
    to = float(np.nansum(d["TO"]))
    checks = float(np.nansum(d["CHECKS"]))
    avg = to / checks if checks else np.nan
    return {"to": to, "checks": checks, "avg": avg}


# =========================
# СБОРКА ОТЧЁТА
# =========================
def build_report(report_date: datetime, rm_filter: Optional[str] = None) -> Tuple[str, Optional[str]]:
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
        msg = "❌ Не хватает файлов:\n" + "\n".join([f"• {os.path.basename(x)}" for x in missing])
        return (msg, None)

    store_rm, store_name = load_roster_maps(path_for("roster", 0))

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

    # ФИЛЬТР ПО РМ (режим "По РМ")
    if rm_filter:
        w26 = w26[w26["rm"].astype(str).str.strip() == str(rm_filter).strip()].copy()
        w25 = w25[w25["rm"].astype(str).str.strip() == str(rm_filter).strip()].copy()

    report_ts = pd.Timestamp(report_date.year, report_date.month, report_date.day)

    # Проверка: введённая дата должна быть в данных 2026 (в выбранном режиме)
    if w26[w26["date"] == report_ts].empty:
        scope = f"для РМ: {rm_filter}" if rm_filter else "по сети"
        msg = (
            "❌ В данных 2026 нет записей за введённую дату.\n"
            f"Контур: {scope}\n"
            f"Дата: {report_date:%d.%m.%y}\n"
            "Проверь, что файлы ТО/чеки/длина 26 загружены и содержат эту дату."
        )
        return (msg, None)

    # Период MTD
    mtd_start_26, mtd_end_26 = period_mtd(report_date)
    mtd_start_25 = pd.Timestamp(mtd_start_26.year - 1, mtd_start_26.month, mtd_start_26.day)
    mtd_end_25 = pd.Timestamp(mtd_end_26.year - 1, mtd_end_26.month, mtd_end_26.day)

    # Метрики MTD (в выбранном контуре)
    net_mtd_26 = network_metrics(w26, mtd_start_26, mtd_end_26)

    # Планы + план на дату + выполнение (в выбранном контуре)
    plans = read_plans(path_for("plans", 0), store_rm)

    stores_in_mtd = set(
        w26[(w26["date"] >= mtd_start_26) & (w26["date"] <= mtd_end_26)]["store_code"].unique()
    )
    plans_used = plans[plans["store_code"].isin(stores_in_mtd)].copy()

    if rm_filter:
        plans_used = plans_used[plans_used["rm"].astype(str).str.strip() == str(rm_filter).strip()].copy()

    month_plan_total = float(np.nansum(plans_used["month_plan"]))
    plan_on_date_total = plan_to_date(month_plan_total, report_date)
    perf_net = (
        net_mtd_26["to"] / plan_on_date_total
        if plan_on_date_total and not np.isnan(plan_on_date_total)
        else np.nan
    )

    # РМ-таблица (для сети: все, для РМ: одна строка)
    if not rm_filter:
        rms_from_roster = {v for v in store_rm.values() if v and str(v).strip()}
        rms_from_plans = {str(x).strip() for x in plans_used["rm"].dropna().astype(str).tolist()}
        rms_from_fact = {
            str(x).strip()
            for x in w26[(w26["date"] >= mtd_start_26) & (w26["date"] <= mtd_end_26)]["rm"].dropna().astype(str).tolist()
        }
        all_rms = sorted({*rms_from_roster, *rms_from_plans, *rms_from_fact})
    else:
        all_rms = [str(rm_filter).strip()]

    fact_by_rm = (
        w26[(w26["date"] >= mtd_start_26) & (w26["date"] <= mtd_end_26)]
        .groupby("rm", as_index=False)["TO"]
        .sum()
        .rename(columns={"TO": "fact"})
    )
    plans_by_rm = plans_used.groupby("rm", as_index=False)["month_plan"].sum()

    rm_tbl = pd.DataFrame({"rm": all_rms})
    rm_tbl = rm_tbl.merge(fact_by_rm, on="rm", how="left").merge(plans_by_rm, on="rm", how="left")
    rm_tbl["plan_on_date"] = rm_tbl["month_plan"].apply(lambda x: plan_to_date(x, report_date))
    rm_tbl["perf"] = rm_tbl["fact"] / rm_tbl["plan_on_date"]
    rm_tbl = rm_tbl.sort_values("perf", ascending=False, na_position="last")

    # LFL MTD (пересечение лавок)
    s26_mtd = per_store_period(w26, mtd_start_26, mtd_end_26).set_index("store_code")
    s25_mtd = per_store_period(w25, mtd_start_25, mtd_end_25).set_index("store_code")
    common = sorted(set(s26_mtd.index).intersection(set(s25_mtd.index)))

    to26 = float(np.nansum(s26_mtd.loc[common, "TO"])) if common else np.nan
    to25 = float(np.nansum(s25_mtd.loc[common, "TO"])) if common else np.nan
    ch26 = float(np.nansum(s26_mtd.loc[common, "CHECKS"])) if common else np.nan
    ch25 = float(np.nansum(s25_mtd.loc[common, "CHECKS"])) if common else np.nan

    avg26 = to26 / ch26 if ch26 else np.nan
    avg25 = to25 / ch25 if ch25 else np.nan

    lfl_to = pct_change(to26, to25)
    lfl_checks = pct_change(ch26, ch25)
    lfl_avg = pct_change(avg26, avg25)

    # Только для сети считаем ТОП/АНТИ-ТОП (для РМ — убираем полностью)
    TOP_N = 5
    top_to = anti_to = top_checks = anti_checks = top_avg = anti_avg = pd.Series(dtype=float)

    lfl_store = pd.DataFrame(index=common)
    if common:
        lfl_store["TO"] = (s26_mtd.loc[common, "TO"] - s25_mtd.loc[common, "TO"]) / s25_mtd.loc[common, "TO"]
        lfl_store["CHECKS"] = (s26_mtd.loc[common, "CHECKS"] - s25_mtd.loc[common, "CHECKS"]) / s25_mtd.loc[common, "CHECKS"]
        lfl_store["AVG"] = (s26_mtd.loc[common, "AVG"] - s25_mtd.loc[common, "AVG"]) / s25_mtd.loc[common, "AVG"]
        lfl_store = lfl_store.replace([np.inf, -np.inf], np.nan)

    if (not rm_filter) and common:
        # ВАЖНО: исключаем выбросы > +50% и < -50% для ТОП/АНТИ-ТОП
        top_to, anti_to = top_anti_n(lfl_store["TO"], TOP_N, -0.5, 0.5)
        top_checks, anti_checks = top_anti_n(lfl_store["CHECKS"], TOP_N, -0.5, 0.5)
        top_avg, anti_avg = top_anti_n(lfl_store["AVG"], TOP_N, -0.5, 0.5)

    # Сколько лавок в плюсе/минусе по LFL Чекам (MTD) — считаем по базе (без обрезки)
    pos_checks = int((lfl_store["CHECKS"] > 0).sum()) if common else 0
    neg_checks = int((lfl_store["CHECKS"] < 0).sum()) if common else 0
    total_lfl_stores = int(lfl_store["CHECKS"].dropna().shape[0]) if common else 0

    # Динамика неделя к неделе | Неделя W vs W-1 (по введённой дате)
    cur_iso_year, cur_week = iso_week_year(report_date)
    prev_iso_year, prev_week = prev_week_of(report_date)

    wk26 = weekly_network(w26, cur_iso_year, cur_week)
    wk26_prev = weekly_network(w26, prev_iso_year, prev_week)

    wow26_to = pct_change(wk26["to"], wk26_prev["to"])
    wow26_checks = pct_change(wk26["checks"], wk26_prev["checks"])
    wow26_avg = pct_change(wk26["avg"], wk26_prev["avg"])

    wk25 = weekly_network(w25, cur_iso_year - 1, cur_week)
    wk25_prev = weekly_network(w25, (prev_iso_year - 1), prev_week)

    wow25_to = pct_change(wk25["to"], wk25_prev["to"])
    wow25_checks = pct_change(wk25["checks"], wk25_prev["checks"])
    wow25_avg = pct_change(wk25["avg"], wk25_prev["avg"])

    # =========================
    # СЛУЖЕБНОЕ СООБЩЕНИЕ
    # =========================
    stores26_mtd = set(s26_mtd.index.astype(str)) if not s26_mtd.empty else set()
    stores25_mtd = set(s25_mtd.index.astype(str)) if not s25_mtd.empty else set()
    excluded_lfl = sorted(list((stores26_mtd ^ stores25_mtd)))  # симм.разность

    wk26_stores = int(w26[(w26["iso_year"] == cur_iso_year) & (w26["iso_week"] == cur_week)]["store_code"].nunique())
    wk26_prev_stores = int(w26[(w26["iso_year"] == prev_iso_year) & (w26["iso_week"] == prev_week)]["store_code"].nunique())
    wk25_stores = int(w25[(w25["iso_year"] == (cur_iso_year - 1)) & (w25["iso_week"] == cur_week)]["store_code"].nunique())
    wk25_prev_stores = int(w25[(w25["iso_year"] == (prev_iso_year - 1)) & (w25["iso_week"] == prev_week)]["store_code"].nunique())

    def _store_label_plain(code_: str) -> str:
        nm_ = store_name.get(code_, "").strip()
        return f"{code_} {nm_}".strip() if nm_ else code_

    extra_lines: List[str] = []
    extra_lines.append(SEP)
    extra_lines.append("ℹ️ <b>СЛУЖЕБНАЯ ИНФОРМАЦИЯ</b>")
    extra_lines.append("")
    extra_lines.append("<b>Исключены из LFL MTD</b> (нет сопоставимых данных 2026↔2025 за период):")
    if excluded_lfl:
        for c in excluded_lfl:
            extra_lines.append(f"• {_store_label_plain(c)}")
    else:
        extra_lines.append("—")
    extra_lines.append("")
    extra_lines.append(f"Исключено: <b>{len(excluded_lfl)}</b> лавок")
    extra_lines.append(f"LFL база: <b>{len(common)}</b> лавок")
    extra_lines.append("")
    extra_lines.append(f"<b>База лавок | Неделя к неделе</b> (Неделя {cur_week} vs {prev_week}):")
    extra_lines.append(f"2026: <b>{wk26_stores}</b> (нед.{cur_week}) / <b>{wk26_prev_stores}</b> (нед.{prev_week})")
    extra_lines.append(f"2025: <b>{wk25_stores}</b> (нед.{cur_week}) / <b>{wk25_prev_stores}</b> (нед.{prev_week})")
    extra_text = "\n".join(extra_lines)

    # ====== СБОРКА ТЕКСТА ======
    period_str = f"{mtd_start_26:%d.%m}–{mtd_end_26:%d.%m}"
    report_date_str = f"{report_date:%d.%m.%y}"
    week_header = f"Неделя {cur_week} vs {prev_week}"

    def _store_label(code: str) -> str:
        nm = store_name.get(code, "").strip()
        return f"{code} {nm}".strip() if nm else code

    lines: List[str] = []

    if rm_filter:
        lines.append(f"👥 <b>АНАЛИТИКА РМ:</b> <b>{rm_filter}</b> | MTD ({period_str})")
    else:
        lines.append(f"📊 <b>АНАЛИТИКА СЕТИ</b> | MTD ({period_str})")

    lines.append(f"Дата отчёта: <b>{report_date_str}</b>")
    lines.append("")
    lines.append(SEP)
    lines.append("")
    lines.append(f"ТО Факт:         <b>{fmt_money(net_mtd_26['to'])} ₽</b>")
    lines.append(f"ТО План на дату: <b>{fmt_money(plan_on_date_total)} ₽</b>")
    lines.append(f"Выполнение плана: <b>{fmt_pct_plain(perf_net)}</b>")
    lines.append("")
    lines.append(f"Чеки:            <b>{fmt_money(net_mtd_26['checks'])}</b>")
    lines.append(f"Ср. чек:         <b>{fmt_money(net_mtd_26['avg'])} ₽</b>")
    lines.append(f"Длина чека:      <b>{fmt_num(net_mtd_26['basket'], 2)}</b>")
    lines.append("")
    lines.append(SEP)
    lines.append("👥 <b>РМ</b> | выполнение плана (MTD)")
    lines.append("")

    if not rm_filter:
        rm_lines = []
        max_name = 0
        for _, r in rm_tbl.iterrows():
            name = str(r["rm"]).strip()
            max_name = max(max_name, len(name))
            rm_lines.append((name, r["perf"]))
        max_name = min(max_name, 28)

        for name, perf in rm_lines:
            n = name[:max_name]
            pad = " " * (max_name - len(n))
            lines.append(f"{n}{pad} — <b>{fmt_pct_plain(perf)}</b>")
    else:
        r = rm_tbl.iloc[0] if not rm_tbl.empty else None
        rm_name = str(rm_filter).strip()
        fact_rm = float(r["fact"]) if r is not None and not pd.isna(r["fact"]) else np.nan
        plan_rm = float(r["plan_on_date"]) if r is not None and not pd.isna(r["plan_on_date"]) else np.nan
        perf_rm = float(r["perf"]) if r is not None and not pd.isna(r["perf"]) else np.nan

        lines.append(f"{rm_name}")
        lines.append(f"ТО: <b>{fmt_money(fact_rm)} ₽</b>   |   План: <b>{fmt_money(plan_rm)} ₽</b>   |   Вып: <b>{fmt_pct_plain(perf_rm)}</b>")

    lines.append("")
    lines.append(SEP)
    lines.append("📈 <b>LFL</b> | MTD (2026 vs 2025)")
    lines.append("")
    lines.append(
        f"ТО: <b>{fmt_pct_signed(lfl_to)}</b>   |   "
        f"Чеки: <b>{fmt_pct_signed(lfl_checks)}</b>   |   "
        f"Ср. чек: <b>{fmt_pct_signed(lfl_avg)}</b>"
    )
    lines.append("")

    # ТОП/АНТИ-ТОП — ТОЛЬКО В РЕЖИМЕ СЕТИ
    if not rm_filter:
        def render_top_anti_block(title: str, top_s: pd.Series, anti_s: pd.Series, n: int = 5):
            lines.append(SEP)
            lines.append(title)
            lines.append("")
            lines.append(f"ТОП-{n}:")
            if top_s is None or len(top_s) == 0:
                lines.append("—")
            else:
                for i, (k, v) in enumerate(top_s.items(), start=1):
                    lines.append(f"{i}) {_store_label(k)}  <b>{fmt_pct_signed(v)}</b>")
            lines.append("")
            lines.append(f"АНТИ-ТОП-{n}:")
            if anti_s is None or len(anti_s) == 0:
                lines.append("—")
            else:
                for i, (k, v) in enumerate(anti_s.items(), start=1):
                    lines.append(f"{i}) {_store_label(k)}  <b>{fmt_pct_signed(v)}</b>")
            lines.append("")

        render_top_anti_block("📊 <b>ТОП / АНТИ-ТОП LFL (MTD) — ТО</b>", top_to, anti_to, TOP_N)
        render_top_anti_block("📊 <b>ТОП / АНТИ-ТОП LFL (MTD) — Чеки</b>", top_checks, anti_checks, TOP_N)
        render_top_anti_block("📊 <b>ТОП / АНТИ-ТОП LFL (MTD) — Ср. чек</b>", top_avg, anti_avg, TOP_N)

    lines.append(SEP)
    lines.append(f"📊 <b>ДИНАМИКА НЕДЕЛЯ К НЕДЕЛЕ</b> | {week_header}")
    lines.append("")
    lines.append("<b>2026:</b>")
    lines.append(
        f"ТО: <b>{fmt_pct_signed(wow26_to)}</b>   |   "
        f"Чеки: <b>{fmt_pct_signed(wow26_checks)}</b>   |   "
        f"Ср. чек: <b>{fmt_pct_signed(wow26_avg)}</b>"
    )
    lines.append("")
    lines.append("<b>2025:</b>")
    lines.append(
        f"ТО: <b>{fmt_pct_signed(wow25_to)}</b>   |   "
        f"Чеки: <b>{fmt_pct_signed(wow25_checks)}</b>   |   "
        f"Ср. чек: <b>{fmt_pct_signed(wow25_avg)}</b>"
    )
    lines.append("")
    lines.append(SEP)
    lines.append("🧠 <b>ВЫВОДЫ</b>")
    lines.append("")
    lines.append(
        f"1) LFL MTD: ТО {fmt_pct_signed(lfl_to)}, Чеки {fmt_pct_signed(lfl_checks)}, Ср. чек {fmt_pct_signed(lfl_avg)} — баланс трафика и среднего чека."
    )
    lines.append(
        f"2) Выполнение плана: {fmt_pct_plain(perf_net)} (план на дату) — при текущем темпе возможен риск недобора."
    )
    if not rm_filter:
        lines.append("3) Фокус — лавки АНТИ-ТОП по LFL: они дают непропорционально большой минус результата.")
        lines.append(
            f"4) Динамика неделя к неделе ({week_header}): 2026 (ТО {fmt_pct_signed(wow26_to)}, Чеки {fmt_pct_signed(wow26_checks)}, Ср. чек {fmt_pct_signed(wow26_avg)}) vs 2025 (ТО {fmt_pct_signed(wow25_to)}, Чеки {fmt_pct_signed(wow25_checks)}, Ср. чек {fmt_pct_signed(wow25_avg)})."
        )
        lines.append(
            f"5) LFL по чекам: в плюсе {pos_checks} лавок, в минусе {neg_checks} лавок (база LFL: {total_lfl_stores})."
        )
    else:
        lines.append(
            f"3) Динамика неделя к неделе ({week_header}): 2026 (ТО {fmt_pct_signed(wow26_to)}, Чеки {fmt_pct_signed(wow26_checks)}, Ср. чек {fmt_pct_signed(wow26_avg)}) vs 2025 (ТО {fmt_pct_signed(wow25_to)}, Чеки {fmt_pct_signed(wow25_checks)}, Ср. чек {fmt_pct_signed(wow25_avg)})."
        )
        lines.append(
            f"4) LFL по чекам: в плюсе {pos_checks} лавок, в минусе {neg_checks} лавок (база LFL: {total_lfl_stores})."
        )

    return ("\n".join(lines), extra_text)


# =========================
# TELEGRAM HANDLERS
# =========================
@bot.message_handler(commands=["start"])
def cmd_start(m):
    WAITING_FOR_REPORT_DATE[m.chat.id] = False
    REPORT_MODE[m.chat.id] = "network"
    SELECTED_RM[m.chat.id] = None

    bot.send_message(
        m.chat.id,
        "Привет! Я Лея 👋\n"
        "Загрузи Excel-файлы (как документы), потом вызови /report.\n\n"
        "Команды:\n"
        "• /files — что загружено\n"
        "• /report — выбрать режим и сформировать отчёт\n"
        "• /version — версия бота"
    )


@bot.message_handler(commands=["version"])
def cmd_version(m):
    bot.send_message(m.chat.id, f"BOT_VERSION: <b>{BOT_VERSION}</b>")


@bot.message_handler(commands=["files"])
def cmd_files(m):
    names = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".xlsx")])
    if not names:
        bot.send_message(m.chat.id, "Файлы не загружены 🙂 Пришли Excel документами.")
        return
    bot.send_message(m.chat.id, "Загружено:\n" + "\n".join([f"• {x}" for x in names]))


@bot.message_handler(commands=["report"])
def cmd_report(m):
    WAITING_FOR_REPORT_DATE[m.chat.id] = False
    SELECTED_RM[m.chat.id] = None

    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(
        types.InlineKeyboardButton("📊 По сети", callback_data="rep|network"),
        types.InlineKeyboardButton("👥 По РМ", callback_data="rep|rm"),
    )
    bot.send_message(m.chat.id, "Выбери тип отчёта:", reply_markup=kb)


@bot.callback_query_handler(func=lambda call: call.data.startswith("rep|"))
def on_report_type(call):
    try:
        chat_id = call.message.chat.id
        mode = call.data.split("|", 1)[1]

        if mode == "network":
            REPORT_MODE[chat_id] = "network"
            SELECTED_RM[chat_id] = None
            WAITING_FOR_REPORT_DATE[chat_id] = True
            bot.answer_callback_query(call.id, "Режим: По сети")
            bot.send_message(
                chat_id,
                "Введи дату для анализа (поддерживаются разные форматы)\n"
                "Примеры: <b>31.01.26</b>, <b>310126</b>, <b>31/01/26</b>, <b>31-01-26</b>"
            )
            return

        if mode == "rm":
            REPORT_MODE[chat_id] = "rm"
            SELECTED_RM[chat_id] = None
            WAITING_FOR_REPORT_DATE[chat_id] = False

            roster_path = path_for("roster", 0)
            if not os.path.exists(roster_path):
                bot.answer_callback_query(call.id, "Нет ростера")
                bot.send_message(chat_id, "❌ Сначала загрузи файл <b>ростер</b>, чтобы я могла показать список РМ.")
                return

            store_rm, _ = load_roster_maps(roster_path)
            rms = sorted({str(v).strip() for v in store_rm.values() if v and str(v).strip()})
            if not rms:
                bot.answer_callback_query(call.id, "РМ не найдены")
                bot.send_message(chat_id, "❌ В ростере не нашла РМ (колонка 'Регион').")
                return

            RM_OPTIONS[chat_id] = rms

            kb = types.InlineKeyboardMarkup(row_width=1)
            for i, rm in enumerate(rms):
                kb.add(types.InlineKeyboardButton(rm, callback_data=f"rm|{i}"))

            bot.answer_callback_query(call.id, "Режим: По РМ")
            bot.send_message(chat_id, "Выбери РМ:", reply_markup=kb)
            return

        bot.answer_callback_query(call.id, "Неизвестный режим")

    except Exception as e:
        try:
            bot.answer_callback_query(call.id, "Ошибка")
        except Exception:
            pass
        bot.send_message(call.message.chat.id, f"❌ Ошибка: {e}")


@bot.callback_query_handler(func=lambda call: call.data.startswith("rm|"))
def on_rm_selected(call):
    try:
        chat_id = call.message.chat.id
        idx = int(call.data.split("|", 1)[1])
        rms = RM_OPTIONS.get(chat_id, [])
        if idx < 0 or idx >= len(rms):
            bot.answer_callback_query(call.id, "РМ не найден")
            bot.send_message(chat_id, "❌ Не нашла выбранного РМ. Попробуй снова: /report")
            return

        SELECTED_RM[chat_id] = rms[idx]
        REPORT_MODE[chat_id] = "rm"
        WAITING_FOR_REPORT_DATE[chat_id] = True

        bot.answer_callback_query(call.id, f"Выбран: {SELECTED_RM[chat_id]}")
        bot.send_message(
            chat_id,
            f"Ок, РМ: <b>{SELECTED_RM[chat_id]}</b>\n\n"
            "Введи дату для анализа (поддерживаются разные форматы)\n"
            "Примеры: <b>31.01.26</b>, <b>310126</b>, <b>31/01/26</b>, <b>31-01-26</b>"
        )

    except Exception as e:
        try:
            bot.answer_callback_query(call.id, "Ошибка")
        except Exception:
            pass
        bot.send_message(call.message.chat.id, f"❌ Ошибка: {e}")


@bot.message_handler(content_types=["document"])
def on_document(m):
    doc = m.document
    kind, year = detect_file_kind(doc.file_name)

    if kind == "unknown":
        bot.send_message(
            m.chat.id,
            "Не поняла тип файла 🤔\n"
            "Назови файл так, чтобы было видно что это:\n"
            "• ТО 25 / ТО 26\n"
            "• чеки 25 / чеки 26\n"
            "• длина 26\n"
            "• планы\n"
            "• ростер"
        )
        return

    save_path = path_for(kind, year)

    file_info = bot.get_file(doc.file_id)
    downloaded = bot.download_file(file_info.file_path)

    with open(save_path, "wb") as f:
        f.write(downloaded)

    bot.send_message(
        m.chat.id,
        f"✅ Сохранила: <b>{os.path.basename(save_path)}</b>\n"
        f"Тип: <b>{kind.upper()}</b>  Год: <b>{year if year else '—'}</b>"
    )


@bot.message_handler(func=lambda msg: True, content_types=["text"])
def on_text(m):
    if WAITING_FOR_REPORT_DATE.get(m.chat.id, False):
        dt = parse_input_date(m.text)
        if not dt:
            bot.send_message(
                m.chat.id,
                "❌ Неверный формат даты.\n"
                "Примеры: <b>31.01.26</b> или <b>310126</b> или <b>31/01/26</b>."
            )
            return

        mode = REPORT_MODE.get(m.chat.id, "network")
        rm_filter = SELECTED_RM.get(m.chat.id)

        if mode == "rm" and not rm_filter:
            WAITING_FOR_REPORT_DATE[m.chat.id] = False
            bot.send_message(m.chat.id, "❌ Не выбран РМ. Напиши /report и выбери режим «По РМ».")
            return

        try:
            main_text, extra_text = build_report(dt, rm_filter=rm_filter if mode == "rm" else None)
        except Exception as e:
            main_text, extra_text = (f"❌ Ошибка при расчёте: {e}", None)

        WAITING_FOR_REPORT_DATE[m.chat.id] = False
        send_long(m.chat.id, main_text)
        if extra_text:
            send_long(m.chat.id, extra_text)
        return

    if m.text.strip().startswith("/"):
        return
    bot.send_message(m.chat.id, "Напиши /report чтобы сделать отчёт 🙂")


if __name__ == "__main__":
    print("Bot is running...")
    bot.infinity_polling(timeout=60, long_polling_timeout=60)
