from datetime import timedelta

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from src.constants import TRAIN_END
from src.objective_alignment import (airing_window, frequency_counts, top_five_target, target_sizes,
    dual_metrics, strength_features, item_neighbors, anchor_scores, blend_scores, merged_airing_segments, retrieval)


def segments():
    return pl.DataFrame({"user_id": ["18443832178892490378"]*3,
        "airing_id": ["a", "a", "b"], "tv_show_id": [1, 1, 2], "channel_id": [10]*3,
        "programme_duration_seconds": [100]*3,
        "observed_start": [TRAIN_END-timedelta(seconds=100), TRAIN_END-timedelta(seconds=40), TRAIN_END],
        "observed_end": [TRAIN_END-timedelta(seconds=50), TRAIN_END-timedelta(seconds=10), TRAIN_END+timedelta(seconds=100)]})


def test_distinct_qualified_airings_and_no_future_seconds():
    airings = airing_window(segments(), TRAIN_END-timedelta(days=1), TRAIN_END)
    assert airings.height == 1
    assert airings["seconds"].to_list() == [80]
    assert frequency_counts(airings)["num_qualifying_broadcasts"].to_list() == [1]
    assert airings["last_observed"][0] < TRAIN_END


def test_target_frequency_ties_ids_and_empty_denominator():
    counts = pl.DataFrame({"user_id": ["u"]*7, "tv_show_id": [8, 3, 7, 1, 2, 9, 0],
                          "num_qualifying_broadcasts": [2, 2, 3, 2, 2, 1, 100], "is_relevant": [True]*7})
    top = top_five_target(counts)
    assert top["tv_show_id"].to_list() == [7, 1, 2, 3, 8]
    assert target_sizes(top, ["u", "empty"])["mean"] == 2.5
    recs = {"u": [7, 1, 2, 3, 8], "empty": [1, 2, 3, 4, 5]}
    metrics = dual_metrics(recs, counts.filter(pl.col("tv_show_id") > 0), top, ["u", "empty"], {"u": [7, 1], "empty": []})
    assert metrics["top5_frequency"]["map5"] == .5
    assert metrics["top5_frequency"]["candidate_recall_micro"] == .4
    assert metrics["top5_frequency"]["candidate_map5_ceiling"] == .2


def test_partial_watches_are_strength_not_positive_targets():
    f = segments().head(1).with_columns(pl.lit("partial").alias("airing_id"))
    airings = airing_window(f, TRAIN_END-timedelta(days=1), TRAIN_END)
    strength = strength_features(airings, TRAIN_END)
    assert strength["n80"][0] == 0
    assert strength["n50"][0] == strength["n20"][0] == 1
    assert strength["meaningful"][0] == 0  # 50% but <60 seconds.
    assert frequency_counts(airings).is_empty()


def test_cosine_neighbors_normalize_popularity_remove_self_and_cap():
    matrix = csr_matrix([[1., 1., 0.], [1., 0., 1.], [1., 0., 0.]])
    neighbors = item_neighbors(matrix, limit=1)
    assert not neighbors.diagonal().any()
    assert np.diff(neighbors.indptr).max() <= 1
    assert neighbors[0, 1] > 0 and neighbors[0, 2] == 0  # Equal strengths break by numeric item order.
    assert np.isclose(neighbors[0, 1], 1/np.sqrt(3))


def test_score_transforms_are_finite_per_user_and_reproducible():
    frame = pl.DataFrame({"user_id": ["u"]*3+["cold"]*3, "tv_show_id": [1, 2, 3]*2,
        "score_exposure_10": [8., 2., 0., 0., 0., 0.], "score_item_neighbors": [1., 2., 3., 0., 0., 0.],
        "score_ppr": [1., .5, .2]*2, "score_context_retrieval": [1., 0., 0., 0., 0., 0.],
        "score_recurrence": [1., 2., 1.]*2})
    anchor = anchor_scores(frame, 10, "personal")
    for transform in ("rank", "exponential"):
        values = blend_scores(frame, np.array([10000., 0., -10000., 1., 1., 1.]), anchor, .2, transform)
        assert np.isfinite(values).all()
        assert np.allclose(values, blend_scores(frame, np.array([10000., 0., -10000., 1., 1., 1.]), anchor, .2, transform))
        assert np.isclose(values[:3].sum(), 1) and np.isclose(values[3:].sum(), 1)


def test_overlap_union_independently_reconstructs_completion(tmp_path):
    start = TRAIN_END-timedelta(seconds=100)
    views = pl.DataFrame({"user_id": ["u", "u"], "channel_id": [1, 1],
        "view_start_utc": [start, start+timedelta(seconds=30)],
        "view_end_utc": [start+timedelta(seconds=60), start+timedelta(seconds=80)], "duraton_seconds": [60, 50]})
    schedule = pl.DataFrame({"airing_id": ["a"], "channel_id": [1], "tv_show_id": [1],
        "programme_start_utc": [start], "programme_end_utc": [TRAIN_END], "programme_duration_seconds": [100]})
    merged = merged_airing_segments(views, schedule, tmp_path/"segments.parquet")
    airings = airing_window(merged, start, TRAIN_END)
    assert airings["seconds"].to_list() == [80]  # Not 110 seconds from overlapping sessions.
    assert frequency_counts(airings)["num_qualifying_broadcasts"].to_list() == [1]
    before = airing_window(merged, start, start+timedelta(seconds=70))
    assert before["coverage"].to_list() == [.7]
    assert frequency_counts(before).is_empty()


def test_exposure_formula_and_cold_fallback_candidate_eligibility():
    airings = airing_window(segments(), TRAIN_END-timedelta(days=1), TRAIN_END)
    strength = strength_features(airings, TRAIN_END)
    user = "18443832178892490378"
    past = pl.DataFrame({"tv_show_id": [1, 2, 3, 4, 5], "channel_id": [10]*5,
                        "programme_start_utc": [TRAIN_END-timedelta(days=1)]*5})
    future = pl.concat([past.with_columns(pl.lit(TRAIN_END).alias("programme_start_utc")),
                        past.head(1).with_columns(pl.lit(TRAIN_END+timedelta(hours=1)).alias("programme_start_utc"))])
    views = pl.DataFrame({"user_id": [user], "channel_id": [10],
                         "view_start_utc": [TRAIN_END-timedelta(seconds=100)], "view_end_utc": [TRAIN_END]})
    ranks = {key: {u: [1, 2, 3, 4, 5] for u in (user, "cold")} for key in ("frequency", "recency", "ppr", "global")}
    pool, frame, checks = retrieval(airings, strength, views, past, future, TRAIN_END,
                                    [user, "cold"], set(range(1, 6)), ranks, pool_size=5)
    one = frame.filter((pl.col("user_id") == user) & (pl.col("tv_show_id") == 1))
    assert np.isclose(one["score_exposure_10"][0], 2/11)
    assert np.isclose(one["score_exposure_80"][0], 2/81)
    assert checks["self_connections"] == 0
    assert all(len(p) == len(set(p)) == 5 and set(p) == set(range(1, 6)) for p in pool.values())
    assert frame.filter(pl.col("user_id") == "cold")["score_exposure_10"].sum() == 0


def test_top5_training_labels_and_lambdarank_groups_are_aligned(monkeypatch):
    import lightgbm as lgb
    from src.reranking import TemporalRanker
    captured = {}

    def capture(self, values, labels, group):
        captured.update(values=values, labels=labels, group=group)
        return self

    monkeypatch.setattr(lgb.LGBMRanker, "fit", capture)
    frame = pl.DataFrame({"user_id": ["z"]*6+["a"]*6, "tv_show_id": list(range(6, 0, -1))*2,
                          "feature": list(range(6, 0, -1))*2})
    counts = pl.DataFrame({"user_id": ["a"]*6+["z"]*6, "tv_show_id": list(range(1, 7))*2,
                           "num_qualifying_broadcasts": [1, 1, 1, 1, 1, 10]+[1]*6})
    TemporalRanker().fit(frame, ["feature"], top_five_target(counts))
    assert captured["group"] == [6, 6]
    assert captured["values"].ravel().tolist() == list(range(1, 7))*2
    assert captured["labels"].tolist() == [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0]
