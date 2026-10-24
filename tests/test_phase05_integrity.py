"""Adversarial tests for graph semantics, fusion, temporal fitting and CSVs."""
from datetime import timedelta
import csv

import numpy as np
import polars as pl
import pytest

from src.baselines import RecencyWeightedRecommender
from src.constants import TRAIN_START, TRAIN_END, TEST_START
from src.personalized_pagerank import PersonalizedPageRankRecommender
from src.optimization import rrf, evaluate
from src.submission import validate_predictions, template_users, write_submission, validate_csv
from src.reranking import candidate_features, AdaptiveHybrid, fit_production_ranker


def history():
    return pl.DataFrame({
        "user_id": ["18443832178892490378", "18443832178892490378", "2", "2", "3"],
        "tv_show_id": [1, 2, 2, 3, 4],
        "is_relevant": [True, True, True, True, False],
        "total_broadcasts_watched": [8, 2, 4, 3, 10000],
        "num_qualifying_broadcasts": [8, 2, 4, 3, 0],
        "total_watched_seconds": [800, 200, 400, 300, 10000],
        "mean_coverage_all": [1.0, 1.0, 1.0, 1.0, 0.1],
        "max_coverage_all": [1.0, 1.0, 1.0, 1.0, 0.1],
        "last_interaction_utc": [TRAIN_END-timedelta(days=1)]*5,
    })


def test_ppr_stationary_equation_and_unit_mass():
    model = PersonalizedPageRankRecommender(tol=1e-9).fit(history(), set(range(1, 7)))
    assert np.allclose(np.asarray(model.transition_matrix.sum(axis=1)).ravel(), 1)
    u = "18443832178892490378"
    x = model.ppr_scores[u]
    restart = np.zeros_like(x)
    restart[len(model.user_ids):] = model.ui[model.user_to_idx[u]].toarray().ravel()
    assert np.abs(x-((1-model.alpha)*model.transition_matrix.T@x + model.alpha*restart)).sum() < 1e-8
    assert np.isclose(x.sum(), 1)
    assert np.allclose(x, model._compute_ppr(u), atol=1e-8)
    assert model.convergence["unconverged_users"] == 0


def test_ppr_rejects_partial_edges_and_keeps_noneligible_graph_context():
    h = history()
    m = PersonalizedPageRankRecommender().fit(h, {1, 3, 5, 6, 7})
    assert 4 not in m.programme_ids  # Only non-positive evidence for item 4.
    assert 2 in m.programme_ids  # Historical node retained even if unavailable later.
    assert "3" not in m.user_ids
    changed = h.with_columns(pl.when(~pl.col("is_relevant")).then(99999999)
                             .otherwise(pl.col("total_broadcasts_watched")).alias("total_broadcasts_watched"))
    m2 = PersonalizedPageRankRecommender().fit(changed, {1, 3, 5, 6, 7})
    assert (m.transition_matrix != m2.transition_matrix).nnz == 0
    assert m.recommend("cold") == m2.recommend("cold")
    assert len(m.recommend("cold")) == 5


def test_ppr_max_iterations_and_determinism():
    h = history()
    m = PersonalizedPageRankRecommender(max_iter=1, tol=1e-15).fit(h, set(range(1, 7)))
    assert m.convergence["max_iterations"] == 1
    assert m.convergence["unconverged_users"] > 0
    m1 = PersonalizedPageRankRecommender().fit(h, set(range(1, 7)))
    m2 = PersonalizedPageRankRecommender().fit(h.reverse(), set(range(1, 7)))
    assert m1.user_rankings == m2.user_rankings


def test_rrf_identity_ties_and_no_duplicate_credit():
    ranks = [{"u": [2, 1, 3, 4, 5]}, {"u": [1, 2, 3, 4, 5]}]
    assert rrf(ranks, [1, 0], ["u"], set(range(1, 7)))["u"] == [2, 1, 3, 4, 5]
    assert rrf(ranks, [.5, .5], ["u"], set(range(1, 7)))["u"][:2] == [1, 2]
    out = rrf([{"u": [0, 1, 1, 99]}], [1], ["u"], set(range(1, 7)))["u"]
    assert len(out) == len(set(out)) == 5
    assert 0 not in out and 99 not in out


@pytest.mark.parametrize("items", [[1, 1, 2, 3, 4], [0, 1, 2, 3, 4], [1, 2], [1, 2, 3, 4, 99], [1.0, 2, 3, 4, 5]])
def test_strict_validator_rejects_bad_items(items):
    with pytest.raises(ValueError):
        validate_predictions({"u": items}, ["u"], set(range(1, 7)))


def test_validator_missing_users_and_exact_large_id_csv(tmp_path):
    with pytest.raises(ValueError):
        validate_predictions({}, ["u"], {1, 2, 3, 4, 5})
    ids = ["18443832178892490378", "000123"]
    template = tmp_path/"template.csv"
    with template.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "tv_show_id"])
        w.writerows((u, "0 0 0 0 0") for u in ids)
    recs = {u: [1, 2, 3, 4, 5] for u in ids}
    output = tmp_path/"submission.csv"
    assert write_submission(output, recs, set(range(1, 6)), template)["valid"]
    assert template_users(output) == ids
    assert validate_csv(output, set(range(1, 6)), template)["rows"] == 2


def test_fixed_cohort_keeps_raw_only_and_cold_submission_users():
    targets = pl.DataFrame({"user_id": ["known"], "tv_show_id": [1], "is_relevant": [True]})
    recs = {u: [1, 2, 3, 4, 5] for u in ["known", "raw_only", "cold"]}
    stats = evaluate("test", recs, targets, ["known", "raw_only"], ["known", "cold"], set(range(1, 6)), 0)
    assert stats["map_at_k_all"] == stats["map_at_k_submission"] == .5
    assert stats["num_evaluation_users_submission"] == 2
    assert stats["num_users_zero_relevant_all"] == 1


def test_recency_final_cutoff_and_eligible_fallback():
    h = history()
    model = RecencyWeightedRecommender(half_life_days=30, cutoff=TEST_START).fit(h, {2, 3, 5, 6, 7})
    assert model.cutoff == TEST_START
    assert set(model.recommend("cold")) <= {2, 3, 5, 6, 7}
    assert len(model.recommend("cold")) == 5


def test_candidate_features_never_accept_target_labels():
    h = history()
    users = ["18443832178892490378", "cold"]
    ranks = {"recency": {u: [1, 2, 3, 4, 5] for u in users}}
    schedule = pl.DataFrame({"tv_show_id": [1, 2, 3, 4, 5], "programme_duration_seconds": [100]*5})
    f1, columns = candidate_features(h, ranks, users, set(range(1, 6)), TRAIN_END, schedule)
    assert "is_relevant" not in columns and "user_id" not in columns and "tv_show_id" not in columns
    assert np.isfinite(f1.select(columns).to_numpy()).all()
    f2, _ = candidate_features(h.reverse(), ranks, users, set(range(1, 6)), TRAIN_END, schedule)
    assert f1.equals(f2)


def test_adaptive_gate_trains_on_supplied_historical_snapshot():
    h = history()
    users = h["user_id"].unique().sort().to_list()
    rankings = {k: {u: [1, 2, 3, 4, 5] for u in users} for k in ("recency", "ppr")}
    gate = AdaptiveHybrid().fit(h, h, users, rankings, set(range(1, 7)))
    output = gate.recommend(h, rankings, users, set(range(1, 7)))
    validate_predictions(output, users, set(range(1, 7)))
    assert sum(gate.usage.values()) == len(users)


def test_production_refit_preserves_snapshot_specific_labels(monkeypatch):
    import lightgbm as lgb

    captured = {}

    def capture_fit(self, values, labels, group):
        captured.update(values=values, labels=labels, group=group)
        return self

    monkeypatch.setattr(lgb.LGBMRanker, "fit", capture_fit)
    user = "18443832178892490378"
    early = pl.DataFrame({"user_id": [user, user], "tv_show_id": [1, 2],
                          "snapshot_feature": [10.0, 11.0]})
    late = early.with_columns((pl.col("snapshot_feature") + 10).alias("snapshot_feature"))
    early_labels = early.filter(pl.col("tv_show_id") == 1).select("user_id", "tv_show_id")
    late_labels = late.filter(pl.col("tv_show_id") == 2).select("user_id", "tv_show_id")
    ranker = fit_production_ranker(early, early_labels, late, late_labels, ["snapshot_feature"])

    # Identical user/item pairs have opposite labels in the two snapshots.
    assert captured["values"].ravel().tolist() == [10.0, 11.0, 20.0, 21.0]
    assert captured["labels"].tolist() == [1, 0, 0, 1]
    assert captured["group"] == [2, 2]
    assert ranker.training_rows == 4
    assert ranker.training_users == 2  # Snapshot-specific queries, not unique raw users.
    assert early["user_id"].to_list() == [user, user]
    assert late["user_id"].to_list() == [user, user]
