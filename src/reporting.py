from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_balance(balance: pd.DataFrame, path: str = "reports/figures/psm_balance.png") -> None:
    pivot = balance.pivot(index="feature", columns="stage", values="smd").sort_values("before")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(pivot["before"], pivot.index, label="Before matching")
    ax.scatter(pivot["after"], pivot.index, label="After matching")
    ax.axvline(0.1, linestyle="--", linewidth=1)
    ax.axvline(-0.1, linestyle="--", linewidth=1)
    ax.set_xlabel("Standardised mean difference")
    ax.set_title("Covariate balance before and after matching")
    ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_event_study(event: pd.DataFrame, path: str = "reports/figures/did_event_study.png") -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(event["week"], event["normalised_difference"], marker="o")
    ax.axvline(-0.5, linestyle="--", linewidth=1)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Week relative to intervention")
    ax.set_ylabel("Treated-control difference, normalised to week -1")
    ax.set_title("Descriptive event-study pattern")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_hte(subgroup: pd.DataFrame, path: str = "reports/figures/hte_subgroups.png") -> None:
    data = subgroup.copy()
    data["group"] = data["engagement_band"].astype(str) + " / " + data["tenure_band"].astype(str)
    data = data.sort_values("estimated_effect")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    xerr = [data["estimated_effect"] - data["ci_low"], data["ci_high"] - data["estimated_effect"]]
    ax.barh(data["group"], data["estimated_effect"], xerr=xerr, capsize=3)
    ax.set_xlabel("Estimated retention probability lift")
    ax.set_title("Exploratory heterogeneous treatment effects")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
