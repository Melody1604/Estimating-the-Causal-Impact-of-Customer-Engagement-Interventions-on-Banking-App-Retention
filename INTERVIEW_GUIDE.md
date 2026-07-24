# Interview Guide

## 90-second explanation

I built a customer-intelligence project around a banking retention intervention. The core business issue was that churn prediction and intervention impact are different questions. I first estimated propensity scores from pre-treatment customer variables and matched treated users to comparable controls. That produced an estimated 3.7-percentage-point improvement in 30-day retention, with a 95% bootstrap interval of roughly 0.9 to 6.9 points. I then used difference-in-differences on eight pre and nine post weeks and estimated about 0.43 additional active days per week.

To support targeting, I created cross-fitted AIPW pseudo-outcomes and an out-of-sample heterogeneous-effect ranking. Low-engagement newer customers showed the strongest estimated lift, although I kept confidence intervals and treated weaker subgroup results cautiously. Separately, I trained an XGBoost churn model with an AUC around 0.72. The important finding was that the top churn-risk group had much lower estimated uplift than the top uplift group, so risk prediction alone would misallocate retention capacity.

I packaged the work with modular Python, dbt, Docker, tests and GitHub Actions. My recommendation would be a randomised holdout and targeting based on incremental effect, cost and customer-contact constraints—not model complexity for its own sake.

## Likely technical questions

### Why use both PSM and DiD?
PSM improves comparability on measured baseline variables for the binary retention outcome. DiD uses longitudinal changes and removes time-invariant customer differences. Agreement across methods strengthens confidence, while disagreement would trigger investigation rather than selective reporting.

### What makes the result causal?
The design and assumptions, not the algorithm. PSM requires conditional exchangeability and overlap. DiD requires parallel trends and no differential concurrent shock. Because the data are observational, I would still validate the decision with randomisation.

### Why not use XGBoost feature importance to explain churn causes?
Feature importance describes predictive contribution within the model. It does not establish what would happen if the feature or an intervention were changed.

### How would you productionise it at a bank?
Use governed warehouse tables, dbt tests, an orchestrator, versioned model artefacts, treatment eligibility rules, holdouts, feature freshness checks, drift and calibration monitoring, privacy controls, and outcome attribution with delayed-label handling.

### How would you decide whether to deploy?
Translate uplift into incremental retained customers and value, subtract intervention and operational costs, apply uncertainty bounds and customer-harm constraints, and compare against a randomised control.
