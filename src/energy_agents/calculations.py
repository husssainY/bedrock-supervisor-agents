"""Deterministic calculations the agents rely on.

Numbers are computed here, in code, and handed to the LLM as facts. The model
explains them; it never does the arithmetic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SlabLine:
    units: float
    rate: float
    amount: float


@dataclass(frozen=True)
class BillBreakdown:
    month: str
    kwh: float
    slabs: list[SlabLine]
    energy_charge: float
    fixed_charge: float
    tax: float
    total: float
    top_slab_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_bill(month: str, kwh: float, tariff: dict) -> BillBreakdown:
    remaining = kwh
    lower = 0.0
    lines: list[SlabLine] = []
    for slab in tariff["slabs"]:
        upper = slab["up_to_kwh"]
        band = remaining if upper is None else min(remaining, upper - lower)
        if band > 0:
            lines.append(SlabLine(units=band, rate=slab["rate"], amount=round(band * slab["rate"], 2)))
            remaining -= band
        if upper is not None:
            lower = upper
        if remaining <= 0:
            break

    energy = round(sum(line.amount for line in lines), 2)
    fixed = float(tariff["fixed_charge"])
    tax = round((energy + fixed) * tariff["tax_rate"], 2)
    return BillBreakdown(
        month=month, kwh=kwh, slabs=lines, energy_charge=energy, fixed_charge=fixed, tax=tax,
        total=round(energy + fixed + tax, 2), top_slab_rate=lines[-1].rate if lines else 0.0,
    )


def usage_summary(monthly: dict[str, float]) -> dict:
    if not monthly:
        return {"months_available": 0}
    months = sorted(monthly)
    latest = months[-1]
    previous = months[-2] if len(months) > 1 else None
    window = months[-4:-1]
    avg_3 = round(sum(monthly[m] for m in window) / len(window), 1) if window else None
    peak = max(months, key=monthly.get)

    summary = {
        "months_available": len(months),
        "latest_month": latest,
        "latest_kwh": monthly[latest],
        "previous_month": previous,
        "previous_kwh": monthly[previous] if previous else None,
        "change_vs_previous_pct": (
            round((monthly[latest] - monthly[previous]) / monthly[previous] * 100, 1) if previous else None
        ),
        "avg_prev_3_months_kwh": avg_3,
        "change_vs_3_month_avg_pct": (
            round((monthly[latest] - avg_3) / avg_3 * 100, 1) if avg_3 else None
        ),
        "peak_month": peak,
        "peak_kwh": monthly[peak],
        "annual_kwh": round(sum(monthly.values()), 1),
    }
    return summary
