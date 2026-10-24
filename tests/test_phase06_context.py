from datetime import timedelta

import numpy as np
import polars as pl

from src.constants import TRAIN_START, TRAIN_END
from src.contextual_reranking import (clip_views, exposure_features, context_features,
                                      cosine_stability, CalibratedAdaptiveBlend)
from src.reranking import candidate_features


def fixture():
    cutoff = TRAIN_END
    users = ["18443832178892490378", "000123"]
    history = pl.DataFrame({
        "user_id": [users[0], users[1]], "tv_show_id": [1, 2], "is_relevant": [True, True],
        "total_broadcasts_watched": [2, 1], "num_qualifying_broadcasts": [2, 1],
        "total_watched_seconds": [7200, 3600], "mean_coverage_all": [1., 1.],
        "max_coverage_all": [1., 1.], "last_interaction_utc": [cutoff-timedelta(days=1)]*2})
    views = pl.DataFrame({"user_id": users, "channel_id": [10, 20],
                          "view_start_utc": [cutoff-timedelta(days=1)]*2,
                          "view_end_utc": [cutoff-timedelta(days=1)+timedelta(hours=1)]*2})
    past = pl.DataFrame({"tv_show_id": [1, 2, 3, 4, 5], "channel_id": [10, 20, 10, 20, 30],
                         "programme_start_utc": [cutoff-timedelta(days=1)]*5,
                         "programme_end_utc": [cutoff-timedelta(days=1)+timedelta(hours=1)]*5,
                         "programme_duration_seconds": [3600]*5})
    future = past.with_columns((pl.col("programme_start_utc")+timedelta(days=7)).alias("programme_start_utc"),
                               (pl.col("programme_end_utc")+timedelta(days=7)).alias("programme_end_utc"))
    ranks = {k: {u: [1, 2, 3, 4, 5] for u in users} for k in ("frequency", "recency")}
    frame, _ = candidate_features(history, ranks, users, set(range(1, 6)), cutoff, future)
    recent = {d: history for d in (7, 14, 28)}
    return frame, history, recent, views, past, future


def test_clip_views_excludes_future_and_clips_crossing_session():
    cutoff = TRAIN_END
    views = pl.DataFrame({"user_id": ["u", "future"], "channel_id": [1, 1],
                          "view_start_utc": [cutoff-timedelta(hours=1), cutoff],
                          "view_end_utc": [cutoff+timedelta(hours=2), cutoff+timedelta(hours=1)]})
    clipped = clip_views(views, cutoff-timedelta(days=1), cutoff)
    assert clipped["user_id"].to_list() == ["u"]
    assert clipped["seconds"].to_list() == [3600]
    assert clipped["view_end_utc"][0] == cutoff


def test_exposure_features_normalize_horizon_and_preserve_baseline():
    frame, history, recent, views, past, future = fixture()
    first, features, profiles = exposure_features(frame, history, recent, views, past, future,
                                                   TRAIN_START, TRAIN_END, TRAIN_END+timedelta(days=14))
    second, _, _ = exposure_features(frame, history, recent, views, past, future,
                                     TRAIN_START, TRAIN_END, TRAIN_END+timedelta(days=28))
    assert first.select(frame.columns).equals(frame)
    assert np.allclose(first["future_airings_per_day"].to_numpy()/2, second["future_airings_per_day"].to_numpy())
    assert np.isfinite(first.select(features).to_numpy()).all()
    assert set(profiles["user_id"]) == set(frame["user_id"])


def test_future_views_cannot_change_exposure_or_context_features():
    frame, history, recent, views, past, future = fixture()
    future_views = views.with_columns((pl.col("view_start_utc")+timedelta(days=2)).alias("view_start_utc"),
                                     (pl.col("view_end_utc")+timedelta(days=2)).alias("view_end_utc"))
    both = pl.concat([views, future_views])
    a, _, _ = exposure_features(frame, history, recent, views, past, future, TRAIN_START, TRAIN_END, TRAIN_END+timedelta(days=14))
    a2, _, _ = exposure_features(frame, history, recent, both, past, future, TRAIN_START, TRAIN_END, TRAIN_END+timedelta(days=14))
    b, features = context_features(a, recent, views, past, future, TRAIN_START, TRAIN_END)
    b2, _ = context_features(a2, recent, both, past, future, TRAIN_START, TRAIN_END)
    assert b.equals(b2)
    row = b.filter((pl.col("user_id") == "18443832178892490378") & (pl.col("tv_show_id") == 1))
    wrong_channel = b.filter((pl.col("user_id") == "18443832178892490378") & (pl.col("tv_show_id") == 2))
    assert row["compatibility_channel_7"][0] == 1
    assert wrong_channel["compatibility_channel_7"][0] == 0
    assert np.isfinite(b.select(features).to_numpy()).all()


def test_preference_stability_distinguishes_identical_and_disjoint_interests():
    recent = pl.DataFrame({"user_id": ["stable", "changed"], "item": [1, 1], "n": [2., 3.]})
    previous = pl.DataFrame({"user_id": ["stable", "changed"], "item": [1, 2], "n": [4., 3.]})
    result = dict(cosine_stability(recent, previous, "item", "n").iter_rows())
    assert np.isclose(result["stable"], 1)
    assert result["changed"] == 0


def test_stability_avoids_integer_overflow_for_watch_seconds():
    profile = pl.DataFrame({"user_id": ["u", "u"], "channel_id": [1, 2], "seconds": [1000000, 2000000]})
    result = cosine_stability(profile, profile, "channel_id", "seconds")
    assert np.isclose(result["stability"][0], 1)


def test_calibration_is_per_user_affine_invariant_and_monotone():
    frame = pl.DataFrame({"user_id": ["a"]*3+["b"]*3, "rank_frequency": [1, 2, 3]*2,
                          "rank_recency": [1, 2, 3]*2})
    scores = np.array([1., 2., 3., 10., 20., 30.])
    blend = CalibratedAdaptiveBlend()
    blend.slope, blend.intercept = 2., -1.
    z = blend.standardized(frame, scores)
    assert np.allclose(z[:3], z[3:], atol=1e-6)
    base, prior = blend.components(frame, scores)
    assert np.all(np.diff(base[:3]) > 0)
    assert np.isclose(base[:3].sum(), 1) and np.isclose(base[3:].sum(), 1)
    assert np.isclose(prior[:3].sum(), 1) and np.isclose(prior[3:].sum(), 1)


def test_context_averages_all_known_future_airings():
    frame, _, recent, views, past, future = fixture()
    extra = future.filter(pl.col("tv_show_id") == 1).with_columns(pl.lit(20, dtype=pl.Int64).alias("channel_id"))
    result, _ = context_features(frame, recent, views, past, pl.concat([future, extra]), TRAIN_START, TRAIN_END)
    row = result.filter((pl.col("user_id") == "18443832178892490378") & (pl.col("tv_show_id") == 1))
    assert np.isclose(row["compatibility_channel_7"][0], .5)


def test_historically_fitted_blend_has_valid_deterministic_cold_user_predictions():
    frame, history, recent, views, past, future = fixture()
    enhanced, _, profiles = exposure_features(frame, history, recent, views, past, future,
                                               TRAIN_START, TRAIN_END, TRAIN_END+timedelta(days=14))
    users = sorted(frame["user_id"].unique().to_list())
    scores = -frame["tv_show_id"].to_numpy().astype(float)
    blend = CalibratedAdaptiveBlend().fit(enhanced, scores, history, profiles, users, set(range(1, 6)))
    recs = blend.recommend(enhanced, scores, profiles.head(0), users, set(range(1, 6)))
    assert recs == blend.recommend(enhanced, scores, profiles.head(0), users, set(range(1, 6)))
    assert sum(blend.usage.values()) == len(users)
    for u in users:
        assert len(recs[u]) == len(set(recs[u])) == 5
        assert set(recs[u]) == set(range(1, 6))
