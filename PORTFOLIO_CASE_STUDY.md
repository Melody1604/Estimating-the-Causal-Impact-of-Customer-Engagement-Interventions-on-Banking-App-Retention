# Portfolio Case Study: Banking App Retention Causal Analysis

## 1. Business problem

A digital bank has launched a personalised in-app engagement intervention. The commercial question is not simply **who is likely to churn**, but:

> Which customers would remain active because they received the intervention?

This distinction matters because a high predicted churn probability does not imply a high incremental response to treatment.

## 2. Data design

The public portfolio version uses a reproducible synthetic dataset with:

- 5,000 customers.
- Eight pre-intervention and nine post-intervention weeks.
- Customer demographics, tenure, balances, digital engagement, products, support contacts, NPS and payment-friction indicators.
- Targeted rather than random treatment assignment, creating observed confounding.
- A binary 30-day retention outcome and weekly activity outcomes.
- Heterogeneous treatment effects embedded in the simulation.

Only pre-treatment variables are used for confounding adjustment, segmentation and model training.

## 3. Causal analysis

### Propensity-score matching

A logistic propensity model estimates each customer's probability of receiving the intervention. Treated customers are matched to comparable untreated customers using nearest-neighbour matching on the logit propensity score with a caliper.

**Estimated ATT on 30-day retention:** **+3.7 percentage points**  
**95% bootstrap CI:** **+0.9 to +6.9 percentage points**  
**Matched treated customers:** **1,874**

After matching, all recorded absolute standardised mean differences are below 0.10, improving comparability on measured covariates.

### Difference-in-differences

Repeated weekly observations are collapsed into customer-level pre/post means. The treated-control difference in changes estimates the intervention's effect on app activity.

**Estimated lift:** **+0.43 weekly active days**  
**95% CI:** **+0.41 to +0.45 days**

The descriptive event-study chart shows relatively stable pre-intervention differences and a clear break at implementation. This supports—but does not prove—the parallel-trends assumption.

### Heterogeneous treatment effects

Cross-fitted nuisance models generate augmented inverse-probability-weighted pseudo-outcomes. A cross-fitted random forest ranks customers by estimated conditional treatment effect, while predefined segment estimates retain confidence intervals.

The strongest predefined group is **low-engagement, new customers**:

- Estimated retention lift: **+17.0 percentage points**
- 95% CI: **+7.1 to +27.0 percentage points**

Several other group intervals cross zero. Those groups should not be automatically targeted without further experimental validation.

## 4. Prediction versus causation

An XGBoost model predicts 30-day churn from pre-treatment customer information.

- ROC AUC: **0.724**
- Average precision: **0.731**
- Brier score: **0.212**

The model is useful for risk forecasting, but it does not estimate the effect of engagement treatment.

A top-20% targeting comparison illustrates the difference:

| Strategy | Average predicted churn | Average estimated uplift |
|---|---:|---:|
| Highest predicted churn | 81.2% | 4.8 pp |
| Highest estimated uplift | 52.6% | 18.4 pp |
| Random benchmark | 54.4% | 5.3 pp |

The highest-risk group is not the most persuadable group. A retention campaign optimised only on churn risk could allocate capacity inefficiently.

## 5. Customer segmentation

Standardised K-means creates four behavioural segments using tenure, balance, engagement, digital share, service contacts, product holdings and prior NPS.

The silhouette score is **0.361**, indicating useful but imperfect separation—more realistic than a perfectly separable synthetic clustering exercise.

Segmentation supports communication and product design; causal uplift remains the preferred basis for treatment allocation.

## 6. Engineering design

The project is packaged as a repeatable workflow:

- Python modules for data generation, causal estimation, segmentation and prediction.
- dbt staging models and a customer-analysis mart using DuckDB.
- Docker packaging.
- Unit tests for data variation and matching execution.
- GitHub Actions for push/pull-request checks and a weekly scheduled refresh.
- Persisted model, metrics, tables and diagnostic charts.

## 7. Commercial recommendation

1. Maintain a randomised holdout before scaling.
2. Prioritise customers by estimated incremental uplift, subject to uncertainty and contact-policy constraints.
3. Do not spend retention budget solely on the highest churn scores.
4. Start with low-engagement, newer customers, where the estimated effect is largest and the confidence interval is positive.
5. Track cost per incremental retained customer, complaint rate, opt-out rate and downstream product engagement.
6. Re-estimate effects after material product, pricing, channel or customer-mix changes.

## 8. Limitations

- The shared data are synthetic; results are a workflow demonstration rather than evidence about a real bank.
- PSM and AIPW depend on no important unmeasured confounding.
- DiD depends on parallel trends and no treatment-specific concurrent shock.
- Heterogeneous effects involve model selection and multiple comparisons.
- A production deployment should include randomisation, treatment-cost data, model monitoring and fairness review.
