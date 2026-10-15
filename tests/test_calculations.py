import pytest

from energy_agents.calculations import compute_bill, usage_summary


@pytest.fixture
def tariff(repo):
    return repo.get_tariff()


def test_bill_spans_all_three_slabs(tariff):
    bill = compute_bill("2026-09", 274, tariff)
    assert [(s.units, s.rate) for s in bill.slabs] == [(100, 4.15), (100, 5.6), (74, 7.15)]
    assert bill.energy_charge == 1504.1
    assert bill.tax == 146.17
    assert bill.total == 1770.27
    assert bill.top_slab_rate == 7.15


def test_bill_within_first_slab(tariff):
    bill = compute_bill("2026-09", 95, tariff)
    assert len(bill.slabs) == 1
    assert bill.energy_charge == pytest.approx(394.25)


def test_zero_usage_still_pays_fixed_charge(tariff):
    bill = compute_bill("2026-09", 0, tariff)
    assert bill.energy_charge == 0
    assert bill.total == pytest.approx(120 * 1.09)


def test_usage_summary(repo):
    summary = usage_summary(repo.get_monthly_usage("C1001"))
    assert summary["latest_month"] == "2026-09"
    assert summary["latest_kwh"] == 274
    assert summary["previous_kwh"] == 199
    assert summary["change_vs_previous_pct"] == 37.7
    assert summary["peak_month"] == "2026-05"


def test_usage_summary_empty():
    assert usage_summary({}) == {"months_available": 0}
