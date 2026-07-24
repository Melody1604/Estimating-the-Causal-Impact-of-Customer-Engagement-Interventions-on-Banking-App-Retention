# Project Charter

## Business question
Does a targeted customer-engagement intervention increase 30-day banking-app retention, and which customers should receive it first?

## Treatment, outcome and unit
- **Unit:** customer.
- **Treatment:** receipt of a personalised in-app engagement intervention at week 0.
- **Primary outcome:** retained in the app during the following 30 days.
- **Secondary outcome:** average weekly active days after release.

## Estimands
- **PSM ATT:** average effect on 30-day retention among customers who received the intervention.
- **DiD effect:** average change in weekly active days for treated customers beyond the contemporaneous change among controls.
- **Exploratory CATE:** differences in expected retention lift across pre-treatment customer profiles.

## Identification assumptions
1. PSM/AIPW require no important unmeasured confounders after conditioning on pre-treatment features.
2. Positivity: comparable treated and control customers exist across the covariate space.
3. SUTVA: one customer's intervention does not materially change another customer's outcome.
4. DiD requires parallel pre-intervention trends and no treatment-specific concurrent shock.
5. Features used for causal adjustment and churn prediction are measured before treatment to avoid leakage.

## Decision rule
Scale the intervention only if the retention lift is commercially meaningful, the confidence interval excludes material harm, pre-trends are credible, and operational capacity exists. Prioritise segments with higher estimated incremental lift rather than simply those with highest predicted churn.
