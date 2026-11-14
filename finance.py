"""
finance.py - Núcleo financiero de DERIVATIVES LAB.

Todo el cálculo vive aquí (sin Streamlit) para poder probarlo de forma independiente
con test_finance.py. Ninguna función sustituye resultados inválidos por ceros: ante
datos incorrectos se lanza FinanceError con un mensaje claro para el usuario.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.optimize import brentq
from scipy.stats import norm


class FinanceError(ValueError):
    """Error de validación o de cálculo financiero (mensaje apto para el usuario)."""


class CurveRangeError(FinanceError):
    """Se pidió un factor de descuento fuera del rango cubierto por la curva."""


# ----------------------------------------------------------------------------
# Convenciones
# ----------------------------------------------------------------------------
DAY_COUNTS = ["ACT/360", "ACT/365F", "30/360", "ACT/ACT"]
BDCS = ["Modified Following", "Following", "Preceding", "Unadjusted"]
FREQ_MONTHS = {"Monthly": 1, "Quarterly": 3, "Semiannual": 6, "Annual": 12}
COMPOUNDINGS = ["Continuous", "Annual", "Semiannual", "Quarterly", "Simple"]
INSTRUMENT_TYPES = ["IRS", "Deposit", "FRA"]
INSTR_COLS = ["Type", "Tenor", "Market Rate (%)", "Start Date", "Maturity Date",
              "Frequency", "Day Count", "Business Day", "Compounding"]


def _blank(v) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() == ""
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def to_date(x) -> date | None:
    """Convierte date/datetime/Timestamp/str ISO a date. Devuelve None si está vacío."""
    if _blank(x):
        return None
    if isinstance(x, pd.Timestamp):
        return x.date()
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    if isinstance(x, str):
        try:
            return date.fromisoformat(x.strip()[:10])
        except ValueError:
            pass
    raise FinanceError(f"Invalid date: {x!r}")


def yearfrac(d1: date, d2: date, conv: str) -> float:
    """Fracción de año entre d1 y d2 según la convención indicada."""
    if d2 < d1:
        raise FinanceError(f"End date {d2} is before start date {d1}.")
    if conv == "ACT/360":
        return (d2 - d1).days / 360.0
    if conv == "ACT/365F":
        return (d2 - d1).days / 365.0
    if conv == "30/360":  # 30/360 US (Bond Basis)
        dd1 = 30 if d1.day == 31 else d1.day
        dd2 = d2.day
        if dd2 == 31 and dd1 == 30:
            dd2 = 30
        return (360 * (d2.year - d1.year) + 30 * (d2.month - d1.month) + (dd2 - dd1)) / 360.0
    if conv == "ACT/ACT":  # ISDA
        if d1 == d2:
            return 0.0
        total = 0.0
        for y in range(d1.year, d2.year + 1):
            start = max(d1, date(y, 1, 1))
            end = min(d2, date(y + 1, 1, 1))
            days_in_year = 366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365
            if end > start:
                total += (end - start).days / days_in_year
        return total
    raise FinanceError(f"Unknown day count convention: {conv}")


_TENOR_RE = re.compile(r"^\s*(\d+)\s*([DWMY])\s*$", re.IGNORECASE)


def parse_tenor(tenor: str) -> tuple[int, str]:
    m = _TENOR_RE.match(str(tenor))
    if not m:
        raise FinanceError(f"Invalid tenor '{tenor}'. Use formats such as 1M, 6M, 1Y, 18M, 30Y.")
    n, unit = int(m.group(1)), m.group(2).upper()
    if n <= 0:
        raise FinanceError(f"Tenor '{tenor}' must be positive.")
    return n, unit


def add_tenor(d: date, tenor: str) -> date:
    n, unit = parse_tenor(tenor)
    delta = {"D": relativedelta(days=n), "W": relativedelta(weeks=n),
             "M": relativedelta(months=n), "Y": relativedelta(years=n)}[unit]
    return d + delta


def adjust(d: date, bdc: str) -> date:
    """Ajuste por día hábil. Solo se consideran fines de semana (no hay calendario de feriados)."""
    if bdc == "Unadjusted" or d.weekday() < 5:
        return d
    one = relativedelta(days=1)
    if bdc == "Following":
        while d.weekday() >= 5:
            d += one
        return d
    if bdc == "Preceding":
        while d.weekday() >= 5:
            d -= one
        return d
    if bdc == "Modified Following":
        f = d
        while f.weekday() >= 5:
            f += one
        if f.month != d.month:
            f = d
            while f.weekday() >= 5:
                f -= one
        return f
    raise FinanceError(f"Unknown business day convention: {bdc}")


def payment_schedule(start: date, maturity: date, freq_months: int, bdc: str) -> list[date]:
    """Fechas [start, p1, ..., pn=maturity]. Generación hacia atrás desde el vencimiento
    (stub corto al inicio si el plazo no es múltiplo de la frecuencia)."""
    if maturity <= start:
        raise FinanceError(f"Maturity {maturity} must be after start {start}.")
    raw = [maturity]
    k = 1
    while True:
        d = maturity - relativedelta(months=freq_months * k)
        if d <= start:
            break
        raw.append(d)
        k += 1
    raw.reverse()
    sched = [start] + [adjust(d, bdc) for d in raw]
    for a, b in zip(sched[:-1], sched[1:]):
        if b <= a:
            raise FinanceError("Payment schedule collapsed after the business-day adjustment; "
                               "check the start/maturity dates and frequency.")
    return sched


# ----------------------------------------------------------------------------
# Tasas <-> factores de descuento
# ----------------------------------------------------------------------------
def df_from_rate(r: float, t: float, comp: str) -> float:
    if comp == "Continuous":
        return math.exp(-r * t)
    if comp == "Simple":
        if 1 + r * t <= 0:
            raise FinanceError("Simple rate gives a non-positive discount factor.")
        return 1.0 / (1 + r * t)
    m = {"Annual": 1, "Semiannual": 2, "Quarterly": 4}.get(comp)
    if m is None:
        raise FinanceError(f"Unknown compounding convention: {comp}")
    if 1 + r / m <= 0:
        raise FinanceError("Rate gives a non-positive discount factor.")
    return (1 + r / m) ** (-m * t)


def rate_from_df(df: float, t: float, comp: str) -> float:
    if df <= 0:
        raise FinanceError("Discount factor must be positive.")
    if t <= 0:
        raise FinanceError("Time must be positive to convert a discount factor into a rate.")
    if comp == "Continuous":
        return -math.log(df) / t
    if comp == "Simple":
        return (1.0 / df - 1.0) / t
    m = {"Annual": 1, "Semiannual": 2, "Quarterly": 4}.get(comp)
    if m is None:
        raise FinanceError(f"Unknown compounding convention: {comp}")
    return m * (df ** (-1.0 / (m * t)) - 1.0)


def validate_rate(r: float, name: str = "Interest rate") -> list[str]:
    """Error si la tasa es imposible; devuelve advertencias si es inusual."""
    if r is None or not np.isfinite(r):
        raise FinanceError(f"{name} is empty or not a number.")
    if r <= -1.0 or r > 1.0:
        raise FinanceError(f"{name} = {r:.2%} is outside the valid range (-100%, 100%].")
    if r < -0.02 or r > 0.25:
        return [f"{name} = {r:.2%} is unusual; check that it is entered in % (e.g. 5 for 5%)."]
    return []


# ----------------------------------------------------------------------------
# Curvas de descuento
# ----------------------------------------------------------------------------
class DiscountCurve:
    """Curva de factores de descuento con interpolación lineal sobre ln(DF).
    El eje de tiempo es ACT/365F desde la fecha de valuación. No extrapola: si se pide
    un plazo mayor al último pilar se lanza CurveRangeError."""

    def __init__(self, val_date: date, times, dfs, pillar_dates=None, name="IRS curve"):
        times = np.asarray(times, dtype=float)
        dfs = np.asarray(dfs, dtype=float)
        if len(times) == 0 or len(times) != len(dfs):
            raise FinanceError("The curve needs at least one pillar.")
        if np.any(dfs <= 0) or not np.all(np.isfinite(dfs)):
            raise FinanceError("The curve contains non-positive or invalid discount factors.")
        if np.any(np.diff(times) <= 0) or times[0] <= 0:
            raise FinanceError("Curve pillars must be strictly increasing and after the valuation date.")
        self.val_date = val_date
        self.name = name
        self.times = np.concatenate([[0.0], times])
        self.dfs = np.concatenate([[1.0], dfs])
        self.pillar_dates = list(pillar_dates) if pillar_dates is not None else None

    @property
    def max_time(self) -> float:
        return float(self.times[-1])

    @property
    def max_date(self) -> date | None:
        return self.pillar_dates[-1] if self.pillar_dates else None

    def df_t(self, t: float) -> float:
        if t < -1e-12:
            raise CurveRangeError("Negative time requested from the curve.")
        if t > self.max_time + 1e-9:
            raise CurveRangeError(
                f"Requested horizon {t:.4f}y exceeds the last curve pillar ({self.max_time:.4f}y). "
                "Extend the IRS curve with a longer maturity; the platform does not extrapolate silently.")
        return float(np.exp(np.interp(max(t, 0.0), self.times, np.log(self.dfs))))

    def df(self, d: date) -> float:
        if d < self.val_date:
            raise CurveRangeError(f"Date {d} is before the curve valuation date {self.val_date}.")
        return self.df_t((d - self.val_date).days / 365.0)


class FlatCurve:
    """Curva plana a partir de una tasa manual (misma interfaz que DiscountCurve)."""

    def __init__(self, val_date: date, rate: float, comp: str = "Continuous", dc: str = "ACT/365F"):
        validate_rate(rate)
        self.val_date, self.rate, self.comp, self.dc = val_date, rate, comp, dc
        self.name = f"Flat {rate:.4%} ({comp}, {dc})"
        self.max_time = float("inf")
        self.max_date = None

    def df_t(self, t: float) -> float:
        return df_from_rate(self.rate, t, self.comp)

    def df(self, d: date) -> float:
        if d < self.val_date:
            raise FinanceError(f"Date {d} is before the valuation date {self.val_date}.")
        return df_from_rate(self.rate, yearfrac(self.val_date, d, self.dc), self.comp)


# ----------------------------------------------------------------------------
# Instrumentos y bootstrapping
# ----------------------------------------------------------------------------
@dataclass
class Instrument:
    kind: str
    rate: float
    start: date
    maturity: date          # ya ajustado por día hábil
    dc: str
    bdc: str
    freq_months: int = 12
    comp: str = "Simple"
    label: str = ""


def instruments_from_table(df: pd.DataFrame, val_date: date) -> list[Instrument]:
    """Convierte la tabla editable en instrumentos validados."""
    out: list[Instrument] = []
    for pos, (_, row) in enumerate(df.iterrows(), start=1):
        vals = [row.get(c) for c in INSTR_COLS]
        if all(_blank(v) for v in vals):
            continue  # fila totalmente vacía: se ignora
        tag = f"Row {pos}"
        kind = row.get("Type")
        if _blank(kind) or kind not in INSTRUMENT_TYPES:
            raise FinanceError(f"{tag}: Instrument Type must be one of {INSTRUMENT_TYPES}.")
        rate_pct = row.get("Market Rate (%)")
        if _blank(rate_pct):
            raise FinanceError(f"{tag} ({kind}): Market Rate is empty.")
        try:
            rate = float(rate_pct) / 100.0
        except (TypeError, ValueError):
            raise FinanceError(f"{tag} ({kind}): Market Rate '{rate_pct}' is not a number.")
        validate_rate(rate, f"{tag} market rate")
        for col in ("Day Count", "Business Day"):
            if _blank(row.get(col)):
                raise FinanceError(f"{tag} ({kind}): {col} is empty.")
        dc, bdc = row["Day Count"], row["Business Day"]
        if dc not in DAY_COUNTS:
            raise FinanceError(f"{tag}: unknown Day Count '{dc}'.")
        if bdc not in BDCS:
            raise FinanceError(f"{tag}: unknown Business Day convention '{bdc}'.")
        start = to_date(row.get("Start Date")) or val_date
        mat_in = to_date(row.get("Maturity Date"))
        tenor = row.get("Tenor")
        if kind == "FRA" and (to_date(row.get("Start Date")) is None or mat_in is None):
            raise FinanceError(f"{tag} (FRA): an FRA needs an explicit Start Date and Maturity Date "
                               "(the start of the forward period and its end). Please fill both.")
        if mat_in is not None:
            maturity = mat_in
        elif not _blank(tenor):
            maturity = add_tenor(start, str(tenor))
        else:
            raise FinanceError(f"{tag} ({kind}): provide a Tenor or a Maturity Date.")
        if start < val_date:
            raise FinanceError(f"{tag}: Start Date {start} is before the valuation date {val_date}.")
        maturity = adjust(maturity, bdc)
        if maturity <= start:
            raise FinanceError(f"{tag}: maturity {maturity} must be after start {start}.")
        freq_m, comp = 12, "Simple"
        if kind == "IRS":
            f = row.get("Frequency")
            if _blank(f) or f not in FREQ_MONTHS:
                raise FinanceError(f"{tag} (IRS): Payment Frequency must be one of {list(FREQ_MONTHS)}.")
            freq_m = FREQ_MONTHS[f]
        else:
            c = row.get("Compounding")
            comp = "Simple" if _blank(c) else c
            if comp not in ("Simple", "Annual", "Continuous"):
                raise FinanceError(f"{tag} ({kind}): Compounding must be Simple, Annual or Continuous.")
        label = str(tenor) if not _blank(tenor) else maturity.isoformat()
        out.append(Instrument(kind, rate, start, maturity, dc, bdc, freq_m, comp, f"{kind} {label}"))
    if not out:
        raise FinanceError("There are no instruments: add at least one market quote.")
    return out


def _interp_partial(times: list[float], dfs: list[float], t: float) -> float:
    if t > times[-1] + 1e-9:
        raise CurveRangeError(
            f"A cash flow at {t:.3f}y is needed before the curve has been built up to that point. "
            "Add a shorter instrument (e.g. a deposit or a swap covering that start date).")
    return float(np.exp(np.interp(t, times, np.log(dfs))))


def _tt(val: date, d: date) -> float:
    return (d - val).days / 365.0


def bootstrap(val_date: date, instruments: list[Instrument]):
    """Bootstrapping secuencial de una sola curva (single-curve).

    - Depósitos / FRA: DF(fin) = DF(inicio) / (1 + r*tau)  (o equivalente con otra capitalización).
    - Swaps: se construye el calendario completo de pagos fijos y se resuelve DF(T_n) tal que
      S * sum(tau_i * DF(t_i)) = DF(inicio) - DF(T_n), interpolando (log-lineal) los DF de los
      pagos intermedios que no son pilares. Se usa un solver (brentq) porque los DF
      intermedios dependen del propio DF(T_n).
    Devuelve (curva, tabla, advertencias)."""
    insts = sorted(instruments, key=lambda i: i.maturity)
    mats = [i.maturity for i in insts]
    dup = {m for m in mats if mats.count(m) > 1}
    if dup:
        raise FinanceError("Duplicate maturities: " + ", ".join(sorted(d.isoformat() for d in dup))
                           + ". Each pillar must have a unique maturity date.")
    times: list[float] = [0.0]
    dfs: list[float] = [1.0]
    warnings: list[str] = []

    for inst in insts:
        t_s = _tt(val_date, inst.start)
        df_s = _interp_partial(times, dfs, t_s)
        t_n = _tt(val_date, inst.maturity)
        if t_n <= times[-1]:
            raise FinanceError(f"{inst.label}: maturity is not beyond the previous pillar.")
        if inst.kind in ("Deposit", "FRA"):
            tau = yearfrac(inst.start, inst.maturity, inst.dc)
            if inst.comp == "Simple":
                x = df_s / (1 + inst.rate * tau)
            elif inst.comp == "Annual":
                x = df_s * (1 + inst.rate) ** (-tau)
            else:
                x = df_s * math.exp(-inst.rate * tau)
        else:
            sched = payment_schedule(inst.start, inst.maturity, inst.freq_months, inst.bdc)
            taus = np.array([yearfrac(a, b, inst.dc) for a, b in zip(sched[:-1], sched[1:])])
            pay_t = np.array([_tt(val_date, d) for d in sched[1:]])
            base_l = np.log(dfs)

            def f(x, pay_t=pay_t, taus=taus, t_n=t_n, df_s=df_s, rate=inst.rate):
                ts = np.append(times, t_n)
                ls = np.append(base_l, math.log(x))
                dfi = np.exp(np.interp(pay_t[:-1], ts, ls))
                annuity = float(np.dot(taus[:-1], dfi) + taus[-1] * x)
                return rate * annuity - (df_s - x)

            lo, hi = 1e-10, 10.0
            if f(lo) * f(hi) > 0:
                raise FinanceError(f"{inst.label}: the bootstrap has no solution (check the quote and conventions).")
            x = brentq(f, lo, hi, xtol=1e-15, rtol=1e-13, maxiter=300)
        if not np.isfinite(x) or x <= 0:
            raise FinanceError(f"{inst.label}: produced a non-positive discount factor ({x}). "
                               "The quote is inconsistent with the previous instruments.")
        if x > dfs[-1] + 1e-12:
            warnings.append(f"{inst.label}: discount factor {x:.6f} is higher than the previous pillar "
                            f"({dfs[-1]:.6f}); the curve implies a negative forward rate. "
                            "Verify the quote if this is not intended.")
        times.append(t_n)
        dfs.append(float(x))

    curve = DiscountCurve(val_date, times[1:], dfs[1:], [i.maturity for i in insts],
                          name="Single-curve IRS bootstrap")
    return curve, insts, warnings


def model_rate(inst: Instrument, curve: DiscountCurve) -> float:
    """Tasa que la curva construida implica para el instrumento (verificación de repricing)."""
    df_s, df_e = curve.df(inst.start), curve.df(inst.maturity)
    if inst.kind in ("Deposit", "FRA"):
        tau = yearfrac(inst.start, inst.maturity, inst.dc)
        ratio = df_s / df_e
        if inst.comp == "Simple":
            return (ratio - 1) / tau
        if inst.comp == "Annual":
            return ratio ** (1 / tau) - 1
        return math.log(ratio) / tau
    sched = payment_schedule(inst.start, inst.maturity, inst.freq_months, inst.bdc)
    annuity = sum(yearfrac(a, b, inst.dc) * curve.df(b) for a, b in zip(sched[:-1], sched[1:]))
    return (df_s - df_e) / annuity


def curve_table(curve: DiscountCurve, insts: list[Instrument], comp_out: str = "Continuous") -> pd.DataFrame:
    rows = []
    prev_t, prev_df = 0.0, 1.0
    for inst, t, d in zip(insts, curve.times[1:], curve.dfs[1:]):
        implied = model_rate(inst, curve)
        rows.append({
            "Instrument": inst.label,
            "Type": inst.kind,
            "Maturity": inst.maturity,
            "Years (ACT/365F)": t,
            "Market Rate": inst.rate,
            "Discount Factor": d,
            "Zero Rate": rate_from_df(d, t, comp_out),
            "Forward Rate": rate_from_df(d / prev_df, t - prev_t, comp_out),
            "Par Rate (repriced)": implied,
            "Repricing Error (bp)": (implied - inst.rate) * 1e4,
        })
        prev_t, prev_df = t, d
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Equity Forward con dividendos discretos
# ----------------------------------------------------------------------------
def clean_dividends(rows, val_date: date, maturity: date) -> list[tuple[date, float]]:
    """rows: iterable de (fecha, monto). Ignora filas totalmente vacías; valida el resto."""
    out = []
    for k, (d, a) in enumerate(rows, start=1):
        if _blank(d) and _blank(a):
            continue
        if _blank(d) or _blank(a):
            raise FinanceError(f"Dividend {k}: both Payment Date and Amount are required.")
        d = to_date(d)
        try:
            a = float(a)
        except (TypeError, ValueError):
            raise FinanceError(f"Dividend {k}: amount '{a}' is not a number.")
        if not np.isfinite(a) or a < 0:
            raise FinanceError(f"Dividend {k}: amount must be zero or positive (got {a}).")
        if not (val_date < d <= maturity):
            raise FinanceError(f"Dividend {k}: payment date {d} is outside the life of the contract "
                               f"({val_date} < date <= {maturity}).")
        out.append((d, a))
    return sorted(out)


class _Rebased:
    """Rebasa una curva a otra fecha de valuación: DF(d) = DF_curva(d)/DF_curva(val)."""

    def __init__(self, base, val_date: date):
        if val_date < base.val_date:
            raise CurveRangeError(f"Valuation date {val_date} is before the curve date {base.val_date}.")
        self.base, self.val_date = base, val_date
        self.ref = base.df(val_date)
        self.name = base.name

    def df(self, d: date) -> float:
        return self.base.df(d) / self.ref


def equity_forward(S0, val_date: date, maturity: date, dividends, disc,
                   delivery_price=None, reval_date=None, reval_spot=None) -> dict:
    """F(0,T) = [S0 - sum D_i * DF(0,t_i)] / DF(0,T)."""
    if S0 is None or not np.isfinite(S0) or S0 <= 0:
        raise FinanceError("Spot price must be positive.")
    if maturity <= val_date:
        raise FinanceError("Maturity date must be after the valuation date (time to maturity > 0).")
    divs = clean_dividends(dividends, val_date, maturity)
    if getattr(disc, "val_date", val_date) != val_date:
        disc = _Rebased(disc, val_date)
    df_T = disc.df(maturity)
    if not (0 < df_T) or not np.isfinite(df_T):
        raise FinanceError("The discount factor at maturity is not valid.")
    rows, pv = [], 0.0
    for k, (d, a) in enumerate(divs, start=1):
        dfi = disc.df(d)
        rows.append({"Dividend #": k, "Payment Date": d, "Years": (d - val_date).days / 365.0,
                     "Dividend Amount": a, "Discount Factor": dfi, "Present Value": a * dfi})
        pv += a * dfi
    F = (S0 - pv) / df_T
    F_nodiv = S0 / df_T
    if F <= 0:
        raise FinanceError(f"PV of dividends ({pv:,.4f}) is not smaller than the spot ({S0:,.4f}): the "
                           "forward would be non-positive. Review the dividend amounts.")
    K = F if delivery_price is None else float(delivery_price)
    value0 = S0 - pv - K * df_T
    res = {"S0": S0, "val_date": val_date, "maturity": maturity, "T": (maturity - val_date).days / 365.0,
           "DF_T": df_T, "PV_divs": pv, "F": F, "F_nodiv": F_nodiv, "K": K, "value0": value0,
           "table": pd.DataFrame(rows, columns=["Dividend #", "Payment Date", "Years", "Dividend Amount",
                                                "Discount Factor", "Present Value"]),
           "reval": None, "curve_name": getattr(disc, "name", "")}
    if reval_date is not None:
        if reval_spot is None or not np.isfinite(reval_spot) or reval_spot <= 0:
            raise FinanceError("Re-valuation requires a positive spot price on that date.")
        if not (val_date < reval_date < maturity):
            raise FinanceError("Re-valuation date must lie strictly between the valuation date and maturity.")
        df_t = disc.df(reval_date)
        rem = [(d, a) for d, a in divs if d > reval_date]
        pv_rem = sum(a * disc.df(d) / df_t for d, a in rem)
        df_tT = df_T / df_t
        fair = (reval_spot - pv_rem) / df_tT
        res["reval"] = {"date": reval_date, "spot": reval_spot, "pv_remaining_divs": pv_rem,
                        "DF_t_T": df_tT, "fair_forward": fair,
                        "value_long": reval_spot - pv_rem - K * df_tT,
                        "n_remaining": len(rem)}
    checks = [
        ("Forward without dividends equals S0 / DF(0,T)", abs(F_nodiv - S0 / df_T) < 1e-10, f"{F_nodiv:.6f}"),
        ("Positive discrete dividends lower the forward price", F <= F_nodiv + 1e-12,
         f"{F:.4f} <= {F_nodiv:.4f}"),
        ("Contract at the fair forward price has zero initial value",
         abs((S0 - pv - F * df_T)) < 1e-9 * max(1, S0), f"{S0 - pv - F * df_T:.2e}"),
    ]
    res["checks"] = checks
    return res


# ----------------------------------------------------------------------------
# Opciones: validaciones comunes
# ----------------------------------------------------------------------------
def validate_option_inputs(S0, K, sigma, T, q=0.0) -> None:
    for name, v in (("Underlying price S0", S0), ("Strike K", K), ("Volatility", sigma),
                    ("Time to maturity", T), ("Dividend yield", q)):
        if v is None or not np.isfinite(v):
            raise FinanceError(f"{name} is empty or not a number.")
    if S0 <= 0:
        raise FinanceError("Underlying price S0 must be positive.")
    if K <= 0:
        raise FinanceError("Strike K must be positive.")
    if sigma <= 0:
        raise FinanceError("Volatility must be strictly positive.")
    if T <= 0:
        raise FinanceError("Time to maturity must be strictly positive.")
    if q < 0 or q >= 1:
        raise FinanceError("Dividend yield must be in [0%, 100%).")


def curve_zero_rate(curve, T: float, t0: float = 0.0) -> float:
    """Tasa cero continua efectiva entre t0 y t0+T implícita en la curva."""
    df = curve.df_t(t0 + T) / curve.df_t(t0)
    return -math.log(df) / T


# ----------------------------------------------------------------------------
# Black-Scholes-Merton
# ----------------------------------------------------------------------------
def bsm(S, K, sigma, T, r, q=0.0, opt="Call") -> dict:
    """BSM con dividend yield continuo. Theta anual y diario (calendario /365);
    Vega y Rho por 1 punto porcentual; también valores por unidad."""
    validate_option_inputs(S, K, sigma, T, q)
    validate_rate(r)
    sq = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sq)
    d2 = d1 - sigma * sq
    eqT, erT = math.exp(-q * T), math.exp(-r * T)
    n1 = norm.pdf(d1)
    call = S * eqT * norm.cdf(d1) - K * erT * norm.cdf(d2)
    put = K * erT * norm.cdf(-d2) - S * eqT * norm.cdf(-d1)
    gamma = eqT * n1 / (S * sigma * sq)
    vega = S * eqT * n1 * sq
    if opt == "Call":
        price = call
        delta = eqT * norm.cdf(d1)
        theta = (-(S * eqT * n1 * sigma) / (2 * sq) + q * S * eqT * norm.cdf(d1) - r * K * erT * norm.cdf(d2))
        rho = K * T * erT * norm.cdf(d2)
        intrinsic = max(S - K, 0.0)
    elif opt == "Put":
        price = put
        delta = -eqT * norm.cdf(-d1)
        theta = (-(S * eqT * n1 * sigma) / (2 * sq) - q * S * eqT * norm.cdf(-d1) + r * K * erT * norm.cdf(-d2))
        rho = -K * T * erT * norm.cdf(-d2)
        intrinsic = max(K - S, 0.0)
    else:
        raise FinanceError("Option type must be Call or Put.")
    return {"d1": d1, "d2": d2, "call": call, "put": put, "price": price, "intrinsic": intrinsic,
            "time_value": price - intrinsic, "delta": delta, "gamma": gamma,
            "theta_annual": theta, "theta_daily": theta / 365.0,
            "vega": vega, "vega_1pct": vega / 100.0, "rho": rho, "rho_1pct": rho / 100.0,
            "r_used": r}


def bsm_with_curve(S, K, sigma, T, curve, q=0.0, opt="Call", t0: float = 0.0) -> dict:
    """BSM descontando con DF de la curva IRS: se usa la tasa cero continua efectiva
    r = -ln(DF(t0,t0+T))/T, de modo que DF(T) y el forward coinciden con la curva."""
    validate_option_inputs(S, K, sigma, T, q)
    r_eff = curve_zero_rate(curve, T, t0)
    out = bsm(S, K, sigma, T, r_eff, q, opt)
    out["r_used"] = r_eff
    return out


# ----------------------------------------------------------------------------
# Árbol binomial CRR
# ----------------------------------------------------------------------------
@dataclass
class TreeResult:
    price: float
    intrinsic: float
    time_value: float
    u: float
    d: float
    dt: float
    p: np.ndarray
    step_df: np.ndarray
    step_rate: np.ndarray
    N: int
    stock: np.ndarray | None = None      # [j (bajadas), i (paso)]
    value: np.ndarray | None = None
    cont: np.ndarray | None = None
    exercised: np.ndarray | None = None
    checks: list = field(default_factory=list)


def crr_option(S0, K, sigma, T, N, opt="Call", style="European", r=None, q=0.0,
               curve=None, t0: float = 0.0, keep_tree: bool = False) -> TreeResult:
    """Árbol CRR: u=exp(sigma*sqrt(dt)), d=1/u, p_i=(g_i-d)/(u-d) con g_i=exp(-q dt)/DF_paso_i.
    Con tasa manual DF_paso=exp(-r dt). Con curva IRS DF_paso_i=DF(t_i+1)/DF(t_i), por lo que
    p y el descuento varían por periodo (el árbol sigue siendo recombinante)."""
    validate_option_inputs(S0, K, sigma, T, q)
    if opt not in ("Call", "Put") or style not in ("European", "American"):
        raise FinanceError("Option type must be Call/Put and exercise style European/American.")
    if not isinstance(N, (int, np.integer)) or N < 1 or N > 5000:
        raise FinanceError("Number of steps must be an integer between 1 and 5000.")
    dt = T / N
    if curve is None:
        if r is None:
            raise FinanceError("A manual interest rate is required.")
        validate_rate(r)
        step_df = np.full(N, math.exp(-r * dt))
    else:
        grid = t0 + dt * np.arange(N + 1)
        d_arr = np.array([curve.df_t(float(t)) for t in grid])
        step_df = d_arr[1:] / d_arr[:-1]
    step_rate = -np.log(step_df) / dt
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(-q * dt) / step_df - d) / (u - d)
    bad = np.where((p <= 0) | (p >= 1))[0]
    if len(bad):
        raise FinanceError(
            f"Risk-neutral probability outside (0,1): p = {p[bad[0]]:.4f} at step {bad[0] + 1} "
            f"(u={u:.4f}, d={d:.4f}). Increase the number of steps, raise the volatility or review r and q "
            "(CRR requires d < exp((r-q)dt) < u).")
    sdt = sigma * math.sqrt(dt)
    sign = 1.0 if opt == "Call" else -1.0

    def payoff(s):
        return np.maximum(sign * (s - K), 0.0)

    j = np.arange(N + 1)
    V = payoff(S0 * np.exp((N - 2 * j) * sdt))
    stock = value = cont = exer = None
    if keep_tree:
        stock = np.full((N + 1, N + 1), np.nan)
        value = np.full((N + 1, N + 1), np.nan)
        cont = np.full((N + 1, N + 1), np.nan)
        exer = np.zeros((N + 1, N + 1), dtype=bool)
        stock[:, N] = S0 * np.exp((N - 2 * j) * sdt)
        value[:, N] = V
    for i in range(N - 1, -1, -1):
        ji = np.arange(i + 1)
        c = step_df[i] * (p[i] * V[:i + 1] + (1 - p[i]) * V[1:i + 2])
        Si = S0 * np.exp((i - 2 * ji) * sdt)
        if style == "American":
            intr = payoff(Si)
            ex = intr > c + 1e-12
            V = np.where(ex, intr, c)
        else:
            V, ex = c, np.zeros(i + 1, dtype=bool)
        if keep_tree:
            stock[:i + 1, i] = Si
            value[:i + 1, i] = V
            cont[:i + 1, i] = c
            exer[:i + 1, i] = ex
    price = float(V[0])
    intrinsic = float(payoff(np.array([S0]))[0])
    return TreeResult(price, intrinsic, price - intrinsic, u, d, dt, p, step_df, step_rate, int(N),
                      stock, value, cont, exer)


# ----------------------------------------------------------------------------
# Tipo de cambio (FX)
# ----------------------------------------------------------------------------
def validate_fx(rate, asset_ccy: str, base_ccy: str) -> None:
    """Convención: rate = unidades de moneda BASE por 1 unidad de moneda del ACTIVO."""
    if rate is None or not np.isfinite(rate) or rate <= 0:
        raise FinanceError("Exchange rate must be a positive number (units of base currency per 1 unit of asset currency).")
    if not asset_ccy or not base_ccy or asset_ccy == base_ccy:
        raise FinanceError("Asset currency and base currency must be different when an exchange rate is used.")


def fx_forward_cip(spot_fx: float, df_asset: float, df_base: float) -> float:
    """Paridad cubierta de tasas: X(0,T) = X0 * DF_activo(0,T) / DF_base(0,T), con X en BASE por ACTIVO.
    Si la tasa de la moneda del activo es mayor que la base, la moneda del activo se deprecia a plazo."""
    if spot_fx <= 0 or df_asset <= 0 or df_base <= 0:
        raise FinanceError("FX spot and discount factors must be positive.")
    return spot_fx * df_asset / df_base


# ----------------------------------------------------------------------------
# Posiciones
# ----------------------------------------------------------------------------
def position_factor(position: str, contracts, multiplier) -> float:
    """+1 (Long) / -1 (Short) multiplicado por contratos y multiplicador."""
    if position not in ("Long", "Short"):
        raise FinanceError("Position type must be Long or Short.")
    for name, v in (("Number of contracts", contracts), ("Contract multiplier", multiplier)):
        if v is None or not np.isfinite(v) or v <= 0:
            raise FinanceError(f"{name} must be a positive number.")
    return (1.0 if position == "Long" else -1.0) * float(contracts) * float(multiplier)


# ----------------------------------------------------------------------------
# Scenario Analysis & P&L Attribution (Delta-Gamma-Theta vs revaluación exacta)
# ----------------------------------------------------------------------------
EXPIRY_TOL = 1e-9


def _option_price(S, K, sigma, T_rem, r, q, opt, curve=None, t0=0.0) -> float:
    """BSM con T_rem > 0; si la opción vence (T_rem ~ 0) usa el payoff (no evalúa BSM con T=0)."""
    if T_rem <= EXPIRY_TOL:
        return max(S - K, 0.0) if opt == "Call" else max(K - S, 0.0)
    if curve is not None:
        return bsm_with_curve(S, K, sigma, T_rem, curve, q, opt, t0)["price"]
    return bsm(S, K, sigma, T_rem, r, q, opt)["price"]


def scenario_analysis(S0, K, sigma, T, r, q, opt, new_S, days, position="Long", contracts=1, multiplier=1,
                      curve=None, t0: float = 0.0) -> dict:
    """dS = S1 - S0; dt = días/365.
    Efectos por unidad (base Long): Delta*dS, 0.5*Gamma*dS^2, Theta_anual*dt.
    Revaluación exacta: BSM con S1, T - dt, mismos K, sigma, r (o curva) y q. Con curva IRS el descuento
    del horizonte restante usa DF(t0+T)/DF(t0+dt), coherente con la curva."""
    validate_option_inputs(S0, K, sigma, T, q)
    if new_S is None or not np.isfinite(new_S) or new_S <= 0:
        raise FinanceError("New stock price must be positive.")
    if days is None or not np.isfinite(days) or days < 0:
        raise FinanceError("Days passed must be zero or positive.")
    total_days = T * 365.0
    if days > total_days + 1e-9:
        raise FinanceError(f"Days passed ({days:g}) exceed the remaining life of the option ({total_days:.2f} days).")
    f = position_factor(position, contracts, multiplier)
    sign = 1.0 if position == "Long" else -1.0
    init = bsm_with_curve(S0, K, sigma, T, curve, q, opt, t0) if curve is not None else bsm(S0, K, sigma, T, r, q, opt)
    dS, dt = new_S - S0, days / 365.0
    T_rem = max(T - dt, 0.0)
    expired = T_rem <= EXPIRY_TOL
    eff = {"Delta": init["delta"] * dS, "Gamma": 0.5 * init["gamma"] * dS ** 2, "Theta": init["theta_annual"] * dt}
    est_unit = sum(eff.values())                      # cambio estimado del precio de la opción (1 opción, base Long)
    new_price = _option_price(new_S, K, sigma, T_rem, r, q, opt, curve, t0 + dt)
    act_unit = new_price - init["price"]
    p0 = init["price"]
    tol = 1e-6 * max(p0, 1.0)
    interpretable = abs(act_unit) > tol
    warnings = []
    if not interpretable:
        warnings.append("Actual P&L is zero or numerically negligible: contribution percentages and the relative error "
                        "are not interpretable and are not shown.")
    if expired:
        warnings.append("The option reaches maturity in this scenario: the exact value is the payoff, not a Black-Scholes price.")
    rows = []
    for name, v in list(eff.items()) + [("Estimated P&L", est_unit), ("Actual P&L", act_unit)]:
        pct = (100.0 if name == "Actual P&L" and interpretable else (v / act_unit * 100.0 if interpretable else float("nan")))
        rows.append({"Component": name, "P&L per unit (position)": sign * v, "P&L Impact (total)": f * v, "Contribution (%)": pct})
    return {"init": init, "dS": dS, "dt": dt, "T_rem": T_rem, "expired": expired, "effects": eff, "factor": f, "sign": sign,
            "initial_price": p0, "est_change": est_unit, "est_new_price": p0 + est_unit, "actual_new_price": new_price,
            "actual_change": act_unit, "est_pnl_total": f * est_unit, "actual_pnl_total": f * act_unit,
            "rel_error_pct": (abs(est_unit - act_unit) / abs(act_unit) * 100.0 if interpretable else None),
            "interpretable": interpretable, "warnings": warnings, "table": pd.DataFrame(rows)}


def scenario_sweep(S0, K, sigma, T, r, q, opt, days, S_grid, curve=None, t0: float = 0.0) -> pd.DataFrame:
    """Precio estimado (Delta-Gamma-Theta) vs exacto para una malla de precios del subyacente."""
    rows = []
    for s in S_grid:
        sc = scenario_analysis(S0, K, sigma, T, r, q, opt, float(s), days, curve=curve, t0=t0)
        rows.append({"New stock price": float(s), "Estimated option price": sc["est_new_price"],
                     "Actual option price": sc["actual_new_price"], "Estimation error": sc["est_new_price"] - sc["actual_new_price"]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Greeks Analysis: barrido de sensibilidades (reutiliza bsm / bsm_with_curve, no agrega fórmulas nuevas)
# ----------------------------------------------------------------------------
GREEK_KEYS = ("price", "delta", "gamma", "theta_annual", "theta_daily", "vega_1pct", "rho_1pct")


def greeks_sweep(S, K, sigma, T, r, q, vary: str, grid, curve=None, t0: float = 0.0) -> pd.DataFrame:
    """Precio y griegas de Call y Put variando UNA variable ('S', 'T', 'sigma' o 'r') sobre `grid`,
    con el resto de parámetros constantes. Columnas: x, <griega>_call, <griega>_put.
    Con curva IRS no se puede variar 'r' (la tasa sale de la curva) y se lanza FinanceError."""
    if vary not in ("S", "T", "sigma", "r"):
        raise FinanceError("vary must be one of 'S', 'T', 'sigma' or 'r'.")
    if vary == "r" and curve is not None:
        raise FinanceError("With the IRS curve the rate comes from the curve, so Rho cannot be swept against a flat rate level.")
    rows = []
    for x in grid:
        a = {"S": S, "T": T, "sigma": sigma, "r": r}
        a[vary] = float(x)
        row = {"x": float(x)}
        for opt in ("Call", "Put"):
            if curve is not None:
                res = bsm_with_curve(a["S"], K, a["sigma"], a["T"], curve, q, opt, t0)
            else:
                res = bsm(a["S"], K, a["sigma"], a["T"], a["r"], q, opt)
            for g in GREEK_KEYS:
                row[f"{g}_{opt.lower()}"] = res[g]
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# IRS: bootstrapping de swaps par (tasas swap -> factores de descuento), base en años
# ----------------------------------------------------------------------------
FREQ_PER_YEAR = {"Annual": 1, "Semiannual": 2, "Quarterly": 4, "Monthly": 12}


def bootstrap_par_swaps(quotes, freq: int = 1) -> dict:
    """Bootstrapping secuencial de factores de descuento a partir de tasas swap par.

    quotes: lista de (vencimiento en años, tasa swap en DECIMAL). freq: pagos por año (1, 2, 4, 12).
    Periodo regular τ = 1/freq (convención de libro de texto: cada periodo mide exactamente 1/freq años).

    Un swap par con tasa fija S_k y pagos en t_1..t_k cumple  S_k·τ·Σ_{i<=k} DF_i + DF_k = 1, de donde
        DF_k = (1 - S_k·τ·Σ_{i<k} DF_i) / (1 + S_k·τ).
    Si un vencimiento de la malla de pagos no tiene cotización, su tasa swap se interpola linealmente entre
    las cotizaciones vecinas (plana fuera del rango) y se marca como 'Interpolated'.
    """
    if freq not in (1, 2, 4, 12):
        raise FinanceError("Payment frequency must be 1, 2, 4 or 12 payments per year.")
    q = []
    for k, item in enumerate(quotes, start=1):
        try:
            T, s = float(item[0]), float(item[1])
        except (TypeError, ValueError, IndexError):
            raise FinanceError(f"Quote {k}: maturity and swap rate must be numbers.")
        if not np.isfinite(T) or T <= 0:
            raise FinanceError(f"Quote {k}: maturity must be a positive number of years.")
        validate_rate(s, f"Quote {k} swap rate")
        n = T * freq
        if abs(n - round(n)) > 1e-9:
            raise FinanceError(f"Quote {k}: maturity {T:g}y is not a multiple of the payment period (1/{freq} y).")
        q.append((round(n) / freq, s))
    if not q:
        raise FinanceError("Enter at least one swap quote.")
    q.sort()
    if len({round(t * freq) for t, _ in q}) != len(q):
        raise FinanceError("Duplicate maturities: each swap maturity must appear only once.")
    qt = np.array([t for t, _ in q])
    qs = np.array([s for _, s in q])
    tau = 1.0 / freq
    n_per = int(round(q[-1][0] * freq))
    quoted = {int(round(t * freq)) for t, _ in q}
    dfs, rows, annuity = [], [], 0.0
    prev_df = 1.0
    for k in range(1, n_per + 1):
        t = k * tau
        S = float(np.interp(t, qt, qs))
        num, den = 1.0 - S * annuity, 1.0 + S * tau
        df = num / den
        if not np.isfinite(df) or df <= 0:
            raise FinanceError(f"Bootstrap failed at t = {t:g}y: the swap rate {S:.4%} gives a non-positive "
                               f"discount factor ({df:.6f}). Review the quotes.")
        zero = df ** (-1.0 / t) - 1.0
        fwd = (prev_df / df - 1.0) / tau
        rows.append({"Period": k, "t (years)": t, "Swap rate S": S, "Source": "Quoted" if k in quoted else "Interpolated",
                     "τ·Σ DF(prior)": annuity, "Numerator 1 − S·τ·ΣDF": num, "Denominator 1 + S·τ": den,
                     "Discount factor DF": df, "Zero rate (annual)": zero, "Forward rate (period, annualised)": fwd,
                     "PV floating interest": prev_df - df})
        annuity += tau * df
        dfs.append(df)
        prev_df = df
    times = np.arange(1, n_per + 1) * tau
    return {"freq": freq, "tau": tau, "times": times, "dfs": np.array(dfs), "quotes": q,
            "table": pd.DataFrame(rows)}


def curve_from_par_bootstrap(boot: dict, val_date: date) -> DiscountCurve:
    """DiscountCurve (interfaz común con el resto de módulos) a partir del bootstrapping por años."""
    pdates = [val_date + relativedelta(days=int(round(t * 365))) for t in boot["times"]]
    return DiscountCurve(val_date, boot["times"], boot["dfs"], pdates, name="Par-swap bootstrap")


def value_swap_par_curve(boot: dict, maturity: float, notional: float = 1.0, fixed_rate=None) -> dict:
    """Valúa un swap de plain vanilla con la curva bootstrapped.

    Pierna flotante (sin nominal):  PV = N·Σ (DF_{i-1} − DF_i) = N·(1 − DF_T)     [cupón proyectado F_i·τ = DF_{i-1}/DF_i − 1]
    Pierna fija (sin nominal):      PV = N·K·τ·Σ DF_i
    Con K = tasa swap par del vencimiento ambas piernas coinciden y el swap vale 0 al inicio.
    """
    if notional is None or not np.isfinite(notional) or notional <= 0:
        raise FinanceError("Notional must be positive.")
    freq, tau = boot["freq"], boot["tau"]
    n = maturity * freq
    if abs(n - round(n)) > 1e-9 or round(n) < 1 or round(n) > len(boot["dfs"]):
        raise FinanceError(f"Swap maturity {maturity:g}y must be a payment date of the curve (multiple of 1/{freq} y, "
                           f"up to {boot['times'][-1]:g}y).")
    n = int(round(n))
    dfs = boot["dfs"][:n]
    df_T = float(dfs[-1])
    annuity = tau * float(dfs.sum())
    par = (1.0 - df_T) / annuity
    K = par if fixed_rate is None else float(fixed_rate)
    validate_rate(K, "Fixed rate")
    prev = np.concatenate([[1.0], dfs[:-1]])
    pv_float_int = notional * (1.0 - df_T)
    pv_fixed_int = notional * K * annuity
    rows = [{"Period": i + 1, "t (years)": (i + 1) * tau, "DF": float(dfs[i]),
             "Projected floating rate": (prev[i] / dfs[i] - 1.0) / tau,
             "Floating payment": notional * (prev[i] / dfs[i] - 1.0),
             "PV floating": notional * (prev[i] - dfs[i]),
             "Fixed payment": notional * K * tau, "PV fixed": notional * K * tau * dfs[i]} for i in range(n)]
    return {"maturity": maturity, "notional": notional, "fixed_rate": K, "par_rate": par, "DF_T": df_T, "annuity": annuity,
            "pv_float_interest": pv_float_int, "pv_float_total": pv_float_int + notional * df_T,
            "pv_fixed_interest": pv_fixed_int, "pv_fixed_total": pv_fixed_int + notional * df_T,
            "value_payer": pv_float_int - pv_fixed_int, "value_receiver": pv_fixed_int - pv_float_int,
            "table": pd.DataFrame(rows)}


# ----------------------------------------------------------------------------
# Equity Forward en años: dividendos DISCRETOS o rendimiento CONTINUO q
# ----------------------------------------------------------------------------
def equity_forward_t(S0, T, disc, mode: str = "discrete", dividends=None, q: float = 0.0,
                     delivery_price=None, reval_t=None, reval_spot=None) -> dict:
    """Forward sobre una acción con tiempos en años (t desde la valuación).

    discrete  : F = (S0 − Σ D_i·DF(t_i)) / DF(T)         (D_i = efectivo pagado en t_i, 0 < t_i <= T)
    continuous: F = S0·e^(−qT) / DF(T)  = S0·e^((r−q)T) si DF(T)=e^(−rT)
    disc debe exponer df_t(t) (FlatCurve o DiscountCurve)."""
    if S0 is None or not np.isfinite(S0) or S0 <= 0:
        raise FinanceError("Spot price must be positive.")
    if T is None or not np.isfinite(T) or T <= 0:
        raise FinanceError("Time to maturity T must be positive (years).")
    if mode not in ("discrete", "continuous"):
        raise FinanceError("Dividend model must be 'discrete' or 'continuous'.")
    df_T = disc.df_t(T)
    if not np.isfinite(df_T) or df_T <= 0:
        raise FinanceError("The discount factor at maturity is not valid.")
    rows, divs, pv = [], [], 0.0
    if mode == "discrete":
        for k, item in enumerate(dividends or [], start=1):
            t, a = item
            if _blank(t) and _blank(a):
                continue
            if _blank(t) or _blank(a):
                raise FinanceError(f"Dividend {k}: both the time and the amount are required.")
            try:
                t, a = float(t), float(a)
            except (TypeError, ValueError):
                raise FinanceError(f"Dividend {k}: time and amount must be numbers.")
            if not np.isfinite(a) or a < 0:
                raise FinanceError(f"Dividend {k}: amount must be zero or positive.")
            if not (0 < t <= T + 1e-12):
                raise FinanceError(f"Dividend {k}: time {t:g}y must satisfy 0 < t <= T = {T:g}y.")
            divs.append((t, a))
        divs.sort()
        for k, (t, a) in enumerate(divs, start=1):
            dfi = disc.df_t(t)
            rows.append({"#": k, "t (years)": t, "Dividend": a, "DF(0,t)": dfi, "PV": a * dfi,
                         "Value at T (D / DF(t,T))": a * dfi / df_T})
            pv += a * dfi
        S_adj = S0 - pv
        if S_adj <= 0:
            raise FinanceError(f"PV of dividends ({pv:,.4f}) is not smaller than the spot ({S0:,.4f}): the forward "
                               "would be non-positive. Review the dividend amounts.")
        q_eq = -math.log(S_adj / S0) / T
    else:
        if q is None or not np.isfinite(q) or q < 0 or q >= 1:
            raise FinanceError("Dividend yield q must be in [0%, 100%).")
        S_adj = S0 * math.exp(-q * T)
        pv = S0 - S_adj
        q_eq = q
    F = S_adj / df_T
    F_nodiv = S0 / df_T
    K = F if delivery_price is None else float(delivery_price)
    value0 = S_adj - K * df_T
    res = {"mode": mode, "S0": S0, "T": T, "q": q if mode == "continuous" else None, "q_equiv": q_eq, "DF_T": df_T,
           "r_cont": -math.log(df_T) / T, "S_adj": S_adj, "PV_divs": pv, "F": F, "F_nodiv": F_nodiv, "K": K,
           "value0": value0, "table": pd.DataFrame(rows, columns=["#", "t (years)", "Dividend", "DF(0,t)", "PV",
                                                                   "Value at T (D / DF(t,T))"]),
           "reval": None, "curve_name": getattr(disc, "name", "")}
    if reval_t is not None:
        if reval_spot is None or not np.isfinite(reval_spot) or reval_spot <= 0:
            raise FinanceError("Re-valuation requires a positive spot price on that date.")
        if not (0 < reval_t < T):
            raise FinanceError("Re-valuation time must lie strictly between 0 and T.")
        df_t = disc.df_t(reval_t)
        df_tT = df_T / df_t
        if mode == "discrete":
            rem = [(t, a) for t, a in divs if t > reval_t]
            pv_rem = sum(a * disc.df_t(t) / df_t for t, a in rem)
            s_adj_t = reval_spot - pv_rem
            n_rem = len(rem)
        else:
            s_adj_t = reval_spot * math.exp(-q * (T - reval_t))
            pv_rem, n_rem = reval_spot - s_adj_t, None
        res["reval"] = {"t": reval_t, "spot": reval_spot, "pv_remaining_divs": pv_rem, "DF_t_T": df_tT,
                        "fair_forward": s_adj_t / df_tT, "value_long": s_adj_t - K * df_tT, "n_remaining": n_rem}
    checks = [("Contract at the fair forward price has zero initial value", abs(S_adj - F * df_T) < 1e-9 * max(1, S0),
               f"{S_adj - F * df_T:.2e}"),
              ("Non-negative dividends lower (or keep) the forward", F <= F_nodiv + 1e-12, f"{F:.4f} <= {F_nodiv:.4f}")]
    if mode == "continuous":
        checks.append(("F = S0·e^(−qT)/DF(T) (equals S0·e^((r−q)T) when DF = e^(−rT))",
                       abs(F - S0 * math.exp(-q * T) / df_T) < 1e-10 * max(1, F), f"{F:.6f}"))
    else:
        checks.append(("F = S0/DF(T) − Σ D_i/DF(t_i,T) (dividends carried to maturity)",
                       abs(F - (F_nodiv - sum(r["Value at T (D / DF(t,T))"] for r in rows))) < 1e-9 * max(1, F), f"{F:.6f}"))
    res["checks"] = checks
    return res
