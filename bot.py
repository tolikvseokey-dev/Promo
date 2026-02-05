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

# Состояние: ждём дату после /report
WAITING_FOR_REPORT_DATE: Dict[int, bool] = {}


# =========================
# УТИЛИТЫ / ФОРМАТЫ
# =========================
SEP = "━━━━━━━━━━━━━━━━━━━━━━"


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
    Ожидаем DD.MM.YY (например 31.01.26).
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
    Метрики НЕ читаем по названиям столбцов.
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
    План — шапка может быть не в первой строке (например, сверху "Примененные фильтры").
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


def top_anti_3(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    top = s.sort_values(ascending=False).head(3)
    anti = s.sort_values(ascending=True).head(3)
    return top, anti


def weekly_network(w: pd.DataFrame, iso_year: int, iso_week: int) -> Dict[str, float]:
    d = w[(w["iso_year"] == iso_year) & (w["iso_week"] == iso_week)]
    to = float(np.nansum(d["TO"]))
    checks = float(np.nansum(d["CHECKS"]))
    avg = to / checks if checks else np.nan
    return {"to": to, "checks": checks, "avg": avg}


# =========================
# СБОРКА ОТЧЁТА (ЭТАЛОН)
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

    report_ts = pd.Timestamp(report_date.year, report_date.month, report_date.day)

    # Проверка: введённая дата должна быть в данных 2026
    if w26[w26["date"] == report_ts].empty:
        return (
            "❌ В данных 2026 нет записей за введённую дату.\n"
            f"Дата: {report_date:%d.%m.%y}\n"
            "Проверь, что файлы ТО/чеки/длина 26 загружены и содержат эту дату."
        )

    # Период MTD
    mtd_start_26, mtd_end_26 = period_mtd(report_date)
    mtd_start_25 = pd.Timestamp(mtd_start_26.year - 1, mtd_start_26.month, mtd_start_26.day)
    mtd_end_25 = pd.Timestamp(mtd_end_26.year - 1, mtd_end_26.month, mtd_end_26.day)

    # Метрики сети MTD (2026)
    net_mtd_26 = network_metrics(w26, mtd_start_26, mtd_end_26)

    # Планы + план на дату + выполнение
    plans = read_plans(path_for("plans", 0), store_rm)

    stores_in_mtd = set(
        w26[(w26["date"] >= mtd_start_26) & (w26["date"] <= mtd_end_26)]["store_code"].unique()
    )
    plans_used = plans[plans["store_code"].isin(stores_in_mtd)].copy()

    month_plan_total = float(np.nansum(plans_used["month_plan"]))
    plan_on_date_total = plan_to_date(month_plan_total, report_date)
    perf_net = (
        net_mtd_26["to"] / plan_on_date_total
        if plan_on_date_total and not np.isnan(plan_on_date_total)
        else np.nan
    )

    # РМ — ВСЕ (из ростера + планов + факта)
    rms_from_roster = {v for v in store_rm.values() if v and str(v).strip()}
    rms_from_plans = {str(x).strip() for x in plans_used["rm"].dropna().astype(str).tolist()}
    rms_from_fact = {
        str(x).strip()
        for x in w26[(w26["date"] >= mtd_start_26) & (w26["date"] <= mtd_end_26)]["rm"].dropna().astype(str).tolist()
    }
    all_rms = sorted({*rms_from_roster, *rms_from_plans, *rms_from_fact})

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

    # ТОП/АНТИ-ТОП LFL (MTD): ТО / Чеки / Ср. чек
    lfl_store = pd.DataFrame(index=common)
    if common:
        lfl_store["TO"] = (s26_mtd.loc[common, "TO"] - s25_mtd.loc[common, "TO"]) / s25_mtd.loc[common, "TO"]
        lfl_store["CHECKS"] = (s26_mtd.loc[common, "CHECKS"] - s25_mtd.loc[common, "CHECKS"]) / s25_mtd.loc[common, "CHECKS"]
        lfl_store["AVG"] = (s26_mtd.loc[common, "AVG"] - s25_mtd.loc[common, "AVG"]) / s25_mtd.loc[common, "AVG"]
        lfl_store = lfl_store.replace([np.inf, -np.inf], np.nan)

    top_to, anti_to = top_anti_3(lfl_store["TO"]) if common else (pd.Series(dtype=float), pd.Series(dtype=float))
    top_checks, anti_checks = top_anti_3(lfl_store["CHECKS"]) if common else (pd.Series(dtype=float), pd.Series(dtype=float))
    top_avg, anti_avg = top_anti_3(lfl_store["AVG"]) if common else (pd.Series(dtype=float), pd.Series(dtype=float))

    # Динамика неделя к неделе | Неделя W vs W-1 (по введённой дате)
    cur_iso_year, cur_week = iso_week_year(report_date)
    prev_iso_year, prev_week = prev_week_of(report_date)

    wk26 = weekly_network(w26, cur_iso_year, cur_week)
    wk26_prev = weekly_network(w26, prev_iso_year, prev_week)

    wow26_to = pct_change(wk26["to"], wk26_prev["to"])
    wow26_checks = pct_change(wk26["checks"], wk26_prev["checks"])
    wow26_avg = pct_change(wk26["avg"], wk26_prev["avg"])

    # 2025: те же номера недель (cur_week vs prev_week) в 2025 году
    wk25 = weekly_network(w25, cur_iso_year - 1, cur_week)
    wk25_prev = weekly_network(w25, (prev_iso_year - 1), prev_week)

    wow25_to = pct_change(wk25["to"], wk25_prev["to"])
    wow25_checks = pct_change(wk25["checks"], wk25_prev["checks"])
    wow25_avg = pct_change(wk25["avg"], wk25_prev["avg"])

    # ====== СБОРКА ТЕКСТА (как в эталоне) ======
    period_str = f"{mtd_start_26:%d.%m}–{mtd_end_26:%d.%m}"
    report_date_str = f"{report_date:%d.%m.%y}"
    week_header = f"Неделя {cur_week} vs {prev_week}"

    # для красивого выравнивания в блоке РМ
    rm_lines = []
    max_name = 0
    for _, r in rm_tbl.iterrows():
        name = str(r["rm"]).strip()
        max_name = max(max_name, len(name))
        rm_lines.append((name, r["perf"]))
    max_name = min(max_name, 28)  # чтобы не раздувало

    def _store_label(code: str) -> str:
        nm = store_name.get(code, "").strip()
        if nm:
            return f"{code} {nm}"
        return code

    lines: List[str] = []
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
    for name, perf in rm_lines:
        n = name[:max_name]
        pad = " " * (max_name - len(n))
        lines.append(f"{n}{pad} — <b>{fmt_pct_plain(perf)}</b>")
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

    def render_top_anti_block(title: str, top_s: pd.Series, anti_s: pd.Series):
        lines.append(SEP)
        lines.append(title)
        lines.append("")
        lines.append("ТОП-3:")
        if top_s is None or len(top_s) == 0:
            lines.append("—")
        else:
            for i, (k, v) in enumerate(top_s.items(), start=1):
                lines.append(f"{i}) {_store_label(k)}  <b>{fmt_pct_signed(v)}</b>")
        lines.append("")
        lines.append("АНТИ-ТОП-3:")
        if anti_s is None or len(anti_s) == 0:
            lines.append("—")
        else:
            for i, (k, v) in enumerate(anti_s.items(), start=1):
                lines.append(f"{i}) {_store_label(k)}  <b>{fmt_pct_signed(v)}</b>")
        lines.append("")

    render_top_anti_block("📊 <b>ТОП / АНТИ-ТОП LFL (MTD) — ТО</b>", top_to, anti_to)
    render_top_anti_block("📊 <b>ТОП / АНТИ-ТОП LFL (MTD) — Чеки</b>", top_checks, anti_checks)
    render_top_anti_block("📊 <b>ТОП / АНТИ-ТОП LFL (MTD) — Ср. чек</b>", top_avg, anti_avg)

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
        f"2) Выполнение плана по сети: {fmt_pct_plain(perf_net)} (план на дату) — при текущем темпе возможен риск недобора."
    )
    lines.append(
        "3) Фокус — лавки АНТИ-ТОП-3 по LFL: они дают непропорционально большой минус сети."
    )
    lines.append(
        f"4) Динамика неделя к неделе ({week_header}): 2026 (ТО {fmt_pct_signed(wow26_to)}, Чеки {fmt_pct_signed(wow26_checks)}, Ср. чек {fmt_pct_signed(wow26_avg)}) vs 2025 (ТО {fmt_pct_signed(wow25_to)}, Чеки {fmt_pct_signed(wow25_checks)}, Ср. чек {fmt_pct_signed(wow25_avg)})."
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
        "• /report — сформировать отчёт"
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
        "Пример: <b>31.01.26</b>"
    )


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
                "Введи дату в формате <b>DD.MM.YY</b>, например <b>31.01.26</b>."
            )
            return

        try:
            text = build_report(dt)
        except Exception as e:
            text = f"❌ Ошибка при расчёте: {e}"

        WAITING_FOR_REPORT_DATE[m.chat.id] = False
        bot.send_message(m.chat.id, text)
        return

    if m.text.strip().startswith("/"):
        return
    bot.send_message(m.chat.id, "Напиши /report чтобы сделать отчёт 🙂")


if __name__ == "__main__":
    print("Bot is running...")
    bot.infinity_polling(timeout=60, long_polling_timeout=60)

