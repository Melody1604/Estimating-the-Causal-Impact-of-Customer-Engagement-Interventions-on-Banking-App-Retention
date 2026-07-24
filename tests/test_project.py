from src.data.generate_synthetic import generate_data
from src.causal.psm import run_psm
from src.pipeline import CAUSAL_FEATURES


def test_generated_data_has_required_variation():
    customers, panel = generate_data(n_customers=800, seed=7)
    assert customers["treated"].nunique() == 2
    assert customers["retained_30d"].nunique() == 2
    assert set(panel["post"].unique()) == {0, 1}
    assert panel["customer_id"].nunique() == 800


def test_psm_runs_and_returns_matches():
    customers, _ = generate_data(n_customers=900, seed=8)
    result = run_psm(customers, CAUSAL_FEATURES, n_bootstrap=50)
    assert result.n_matched > 100
    assert -1 <= result.att <= 1
    assert result.ci_low <= result.att <= result.ci_high
