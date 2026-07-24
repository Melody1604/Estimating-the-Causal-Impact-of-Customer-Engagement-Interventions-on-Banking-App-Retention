# Estimating the Causal Impact of Customer Engagement Interventions on Banking App Retention

An end-to-end applied-science portfolio project that separates **causal impact estimation** from **churn prediction**, then packages both into a reproducible analytics pipeline.

## Executive summary

A digital bank wants to know whether a personalised in-app engagement intervention genuinely improves 30-day retention—not merely whether treated customers look different. This project combines:

- Propensity-score matching for the ATT on retention.
- Difference-in-differences for weekly active days.
- Confidence intervals, balance diagnostics and pre-trend checks.
- Exploratory heterogeneous treatment effects using cross-fitted AIPW pseudo-outcomes.
- K-means customer segmentation.
- XGBoost churn prediction, explicitly separated from causal estimation.
- A dbt transformation layer, Docker packaging and GitHub Actions CI/scheduling.

The repository uses synthetic data so it can be shared publicly without exposing customer information. The data-generating process intentionally includes targeted treatment assignment, observed confounding and heterogeneous effects.

## Headline results from the reproducible run

- PSM ATT on 30-day retention: **+3.7 percentage points** (95% CI: **+0.9 to +6.9 pp**).
- Difference-in-differences: **+0.43 weekly active days** (95% CI: **+0.41 to +0.45**).
- Highest-value predefined subgroup: **low-engagement, new customers**, estimated lift **+17.0 pp** (95% CI: **+7.1 to +27.0 pp**).
- XGBoost churn prediction: **ROC AUC 0.724** and **average precision 0.731**.
- Top-20% high-uplift customers show substantially greater estimated incremental lift than the top-20% highest-risk customers, illustrating why prediction and causation must be separated.

For a recruiter-friendly narrative, see [`PORTFOLIO_CASE_STUDY.md`](PORTFOLIO_CASE_STUDY.md). For interview preparation, see [`INTERVIEW_GUIDE.md`](INTERVIEW_GUIDE.md).

## Business framing

**Question:** Does the intervention cause higher retention, and which customers create the largest incremental return?

**Why prediction alone is insufficient:** A churn model can identify who is likely to leave, but it cannot tell us who will stay because of an intervention. High-risk customers may be difficult to influence, while medium-risk customers may be highly persuadable.

See [`PROJECT_CHARTER.md`](PROJECT_CHARTER.md) for estimands, assumptions and the decision rule.

## Repository structure

```text
src/data/            synthetic customer and panel data
src/causal/          PSM, DiD and heterogeneous-effect estimation
src/segmentation/    K-means customer segmentation
src/models/          XGBoost churn prediction
src/pipeline.py      end-to-end orchestration
dbt/                 staging and customer-analysis marts
.github/workflows/   CI plus scheduled refresh
reports/             metrics, tables and diagnostic charts
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python -m src.data.generate_synthetic
python -m src.pipeline
pytest -q
```

Optional dbt layer:

```bash
dbt seed --project-dir dbt --profiles-dir dbt
dbt run --project-dir dbt --profiles-dir dbt
dbt test --project-dir dbt --profiles-dir dbt
```

Docker:

```bash
docker build -t banking-retention-causal .
docker run --rm banking-retention-causal
```

## Outputs

After running the pipeline:

- `reports/causal_customer_intelligence_report.html` — self-contained rendered case-study report.
- `reports/metrics.json` — headline estimates and model metrics.
- `reports/tables/psm_balance.csv` — before/after standardised mean differences.
- `reports/figures/psm_balance.png` — matching balance diagnostic.
- `reports/figures/did_event_study.png` — descriptive pre/post trend diagnostic.
- `reports/tables/hte_subgroups.csv` — segment-level estimated retention lift.
- `reports/tables/segment_profiles.csv` — customer segment profiles.
- `reports/tables/churn_feature_importance.csv` — predictive importance only.

## Methodological guardrails

- PSM estimates are conditional on measured pre-treatment covariates and are not immune to hidden confounding.
- The DiD result depends on credible parallel trends and absence of treated-group-specific shocks.
- Heterogeneous effects are exploratory and should be validated in a randomised rollout.
- XGBoost predicts churn risk; feature importance must not be presented as causal evidence.
- Synthetic results demonstrate workflow capability, not a real bank's commercial outcome.

## Commercial recommendation framework

1. Validate uplift with a randomised holdout before full rollout.
2. Target customers by **incremental treatment effect**, not churn probability alone.
3. Exclude or redesign outreach for segments with low or uncertain lift.
4. Track retention lift, complaint rate, opt-out rate and cost per incremental retained customer.
5. Re-estimate effects after product, pricing or channel changes.

## CV-ready project bullet

> Built an end-to-end causal customer-intelligence pipeline to estimate the impact of a banking-app engagement intervention on retention, combining propensity-score matching, fixed-effects difference-in-differences, cross-fitted heterogeneous treatment effects, customer segmentation and XGBoost churn prediction; productionised the workflow with Python, SQL/dbt, Docker, automated tests and GitHub Actions.
