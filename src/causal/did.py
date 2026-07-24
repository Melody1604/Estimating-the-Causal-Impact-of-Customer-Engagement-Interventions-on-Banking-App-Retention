"""Difference-in-differences for repeated customer activity observations."""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import statsmodels.api as sm


@dataclass
class DIDResult:
    estimate: float
    std_error: float
    ci_low: float
    ci_high: float
    p_value: float
    event_study: pd.DataFrame


def run_did(panel: pd.DataFrame, outcome: str = "weekly_active_days") -> DIDResult:
    # In a two-group, two-period design, first-differencing removes all
    # time-invariant customer characteristics. We collapse repeated weeks into
    # pre/post means and estimate the treated-control difference in changes.
    periods = (
        panel.groupby(["customer_id", "treated", "post"], as_index=False)[outcome]
        .mean()
        .pivot(index=["customer_id", "treated"], columns="post", values=outcome)
        .rename(columns={0: "pre", 1: "post"})
        .reset_index()
    )
    periods["change"] = periods["post"] - periods["pre"]
    X = sm.add_constant(periods[["treated"]])
    model = sm.OLS(periods["change"], X).fit(cov_type="HC3")

    term = "treated"
    estimate = float(model.params[term])
    se = float(model.bse[term])
    ci = model.conf_int().loc[term]
    p_value = float(model.pvalues[term])

    # Descriptive event-study differences, normalised to week -1.
    means = panel.groupby(["week", "treated"], as_index=False)[outcome].mean()
    wide = means.pivot(index="week", columns="treated", values=outcome).rename(columns={0: "control", 1: "treated"})
    wide["difference"] = wide["treated"] - wide["control"]
    baseline = wide.loc[-1, "difference"]
    wide["normalised_difference"] = wide["difference"] - baseline
    event_study = wide.reset_index()

    return DIDResult(estimate, se, float(ci.iloc[0]), float(ci.iloc[1]), p_value, event_study)
