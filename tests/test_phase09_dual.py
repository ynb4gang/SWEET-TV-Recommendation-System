from datetime import timedelta

import numpy as np
import polars as pl
import pytest

from src.constants import TRAIN_START, TRAIN_END
from src.dual_rankers import (PARAMETERS, margin_probabilities, channel_hour_gate,
                              fuse_probabilities, specialist_features)
from src.objective_alignment import stable_rank, positive_candidate_union
from src.reranking import TemporalRanker


def test_personal_ranking_rejects_zero_negative_and_nonfinite():
    ids = np.array([1, 2, 3, 4, 5, 6])
    values = np.array([0., np.nan, -1., 2., 2., np.inf])
    assert stable_rank(values, ids, 100).tolist() == [3, 4]


def test_personal_union_deduplicates_and_preserves_collaborative_evidence():
    ids = np.array([1, 2, 3, 4, 5])
    chosen, owners, _ = positive_candidate_union(
        {"personal": np.array([0., 2., 0., 0., 0.]), "neighbors": np.array([0., 1., 0., 0., 3.])},
        ids, np.array([3, 0, 2, 1, 4]), pool_size=4)
    assert chosen == [2, 5, 4, 1]
    assert owners == {2: "personal", 5: "neighbors", 4: "global_fallback", 1: "global_fallback"}
    assert len(chosen) == len(set(chosen))


def test_zero_sources_use_popularity_not_arbitrary_numeric_ties():
    ids = np.array([1, 2, 3, 4, 5])
    chosen, owners, _ = positive_candidate_union({"empty": np.zeros(5)}, ids, np.array([4, 3, 2, 1, 0]), pool_size=3)
    assert chosen == [5, 4, 3]
    assert set(owners.values()) == {"global_fallback"}


def test_user_softmax_matches_requested_transform_and_handles_extreme_margins():
    frame = pl.DataFrame({"user_id": ["a", "a", "b", "b"]})
    margin = np.array([10000., 9999., -10000., -10001.])
    p = margin_probabilities(frame, margin)
    assert np.allclose(p, [1/(1+np.exp(-1)), 1/(1+np.exp(1))]*2)
    assert np.isfinite(p).all() and np.isclose(p[:2].sum(), 1) and np.isclose(p[2:].sum(), 1)
    with pytest.raises(ValueError): margin_probabilities(frame, margin[:2])


def test_channel_hour_gate_uses_only_past_and_is_bounded():
    views = pl.DataFrame({"user_id": ["stable", "stable", "changed", "changed"], "channel_id": [1, 1, 1, 2],
        "view_start_utc": [TRAIN_END-timedelta(days=d) for d in (14, 1, 14, 1)],
        "view_end_utc": [TRAIN_END-timedelta(days=d)+timedelta(seconds=s) for d,s in ((14,1000),(1,100),(14,1000),(1,100))]})
    users = ["stable", "changed", "cold"]
    before = channel_hour_gate(views, users, TRAIN_END)
    future = views.with_columns((pl.col("view_start_utc")+timedelta(days=40)).alias("view_start_utc"),
                                (pl.col("view_end_utc")+timedelta(days=40)).alias("view_end_utc"))
    assert before.equals(channel_hour_gate(pl.concat([views, future]), users, TRAIN_END))
    assert np.isclose(before["recent_gate"][0], 1)
    assert before["recent_gate"][1:].to_list() == [0., 0.]


def test_fusion_preserves_alignment_and_rejects_duplicate_or_missing_gates():
    frame = pl.DataFrame({"user_id": ["b", "a", "b", "a"]})
    a = np.array([.9, .8, .1, .2]); b = 1-a
    gate = pl.DataFrame({"user_id": ["a", "b"], "recent_gate": [1., 0.]})
    assert np.allclose(fuse_probabilities(frame, a, b, gate), [.1, .8, .9, .2])
    with pytest.raises(ValueError): fuse_probabilities(frame, a, b[:1])
    with pytest.raises(ValueError): fuse_probabilities(frame, a, b, gate.head(1))
    with pytest.raises(ValueError): fuse_probabilities(frame, a, b, pl.concat([gate, gate]))


def test_specialists_explicitly_normalize_logged_exposure_and_have_distinct_views():
    columns = ("rank_global rank_frequency rank_recency rank_ppr rank_cbf rank_pool frequency_share mean_coverage_all max_coverage_all "
        "observed_days_ago normalized_score_item_neighbors normalized_score_ppr normalized_score_frequency normalized_score_recency "
        "active_day_fraction frequency_per_active_day compatibility_channel_7 compatibility_hour_7 compatibility_dow_7 "
        "compatibility_channel_hour_7 duration_mismatch_7 programme_rate_7 programme_active_rate_7 programme_exposure_rate_7 "
        "activity_shift_7 programme_shift_7 active_fraction_7 total_broadcasts_watched num_qualifying_broadcasts recency_frequency global_users "
        "n20 n50 n80 n300 meaningful strength_seconds user_frequency global_frequency score_exposure_10 score_exposure_80 "
        "score_recurrence qualified_14 qualified_28 historical_airings_per_day future_airings_per_day airing_rate_shift "
        "consumption_per_opportunity completion_per_opportunity programme_rate_14 programme_rate_28 programme_shift_28 "
        "activity_shift_28 duration_mismatch_28 programme_stability channel_stability").split()
    frame = pl.DataFrame({"user_id": ["u"], "tv_show_id": [1], **{c: [1.] for c in columns}})
    days = (TRAIN_END-TRAIN_START).total_seconds()/86400
    frame = frame.with_columns(pl.lit(np.log1p(2*days)).alias("n80"), pl.lit(70.).alias("score_exposure_10"))
    views = pl.DataFrame({"user_id": ["u"], "channel_id": [1], "view_start_utc": [TRAIN_END-timedelta(days=1)],
                           "view_end_utc": [TRAIN_END-timedelta(days=1)+timedelta(hours=1)]})
    future = pl.DataFrame({"tv_show_id": [1], "channel_id": [1], "programme_start_utc": [TRAIN_END]})
    result, sets = specialist_features(frame, views, future, TRAIN_END, TRAIN_END+timedelta(days=35))
    assert np.isclose(result["history_rate_n80"][0], np.log1p(2))
    assert np.isclose(result["future_rate_score_exposure_10"][0], np.log1p(2))
    assert "compatibility_dow_hour_7" in sets["A_recent"] and "compatibility_dow_hour_7" not in sets["B_horizon"]
    assert "history_rate_n80" in sets["B_horizon"] and "history_rate_n80" not in sets["A_recent"]
    assert np.isfinite(result.select(list(dict.fromkeys(sets["A_recent"]+sets["B_horizon"]))).to_numpy()).all()


def test_ranker_parameter_override_preserves_legacy_defaults(monkeypatch):
    import lightgbm as lgb
    monkeypatch.setattr(lgb.LGBMRanker, "fit", lambda self, *args, **kwargs: self)
    frame = pl.DataFrame({"user_id": ["u", "u"], "tv_show_id": [1, 2], "x": [1., 0.]})
    truth = frame.select("user_id", "tv_show_id").with_columns(pl.Series("is_relevant", [True, False]))
    legacy = TemporalRanker().fit(frame, ["x"], truth)
    specialist = TemporalRanker(PARAMETERS).fit(frame, ["x"], truth)
    assert legacy.model.get_params()["n_estimators"] == 100
    assert specialist.model.get_params()["n_estimators"] == 350
    assert specialist.model.get_params()["num_leaves"] == 31
    assert specialist.model.get_params()["reg_lambda"] == 10
