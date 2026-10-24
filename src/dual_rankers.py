"""Complementary fixed LambdaRank feature views and cutoff-only stability fusion."""
from datetime import timedelta

import numpy as np
import polars as pl

from .constants import TRAIN_START
from .contextual_reranking import clip_views, _distribution


PARAMETERS = dict(num_leaves=31, n_estimators=350, learning_rate=.045,
                  min_child_samples=60, reg_lambda=10.)


def compatibility(frame, observed, future, keys):
    users = sorted(frame["user_id"].unique().to_list())
    items = sorted(frame["tv_show_id"].unique().to_list())
    slots = {tuple(r): i for i, r in enumerate(pl.concat([observed.select(keys), future.select(keys)]).unique().sort(keys).iter_rows())}
    up = _distribution(observed, "user_id", keys, "seconds", users, slots)
    ip = _distribution(future, "tv_show_id", keys, "weight", items, slots)
    ui, ii = {u: i for i, u in enumerate(users)}, {p: i for i, p in enumerate(items)}
    ur = np.array([ui[u] for u in frame["user_id"]])
    ir = np.array([ii[p] for p in frame["tv_show_id"]])
    values = np.zeros(frame.height, dtype=np.float32)
    for lo in range(0, len(users), 128):
        indices = np.flatnonzero((ur >= lo) & (ur < lo+128))
        dense = (up[lo:lo+128] @ ip.T).toarray()
        values[indices] = dense[ur[indices]-lo, ir[indices]]
    return values


def channel_hour_gate(views, users, cutoff):
    """Hypothesis: confidence in recent ranker = clip(2*(cos(last7,last28)-.5),0,1)."""
    v28 = clip_views(views, max(TRAIN_START, cutoff-timedelta(days=28)), cutoff)
    v7 = clip_views(views, max(TRAIN_START, cutoff-timedelta(days=7)), cutoff)
    keys = ["channel_id", "hour"]
    slots = {tuple(r): i for i, r in enumerate(v28.select(keys).unique().sort(keys).iter_rows())}
    a = _distribution(v7, "user_id", keys, "seconds", users, slots)
    b = _distribution(v28, "user_id", keys, "seconds", users, slots)
    dot = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    norm = np.sqrt(np.asarray(a.multiply(a).sum(axis=1)).ravel() * np.asarray(b.multiply(b).sum(axis=1)).ravel())
    cosine = np.clip(dot/np.maximum(norm, 1e-12), 0, 1)
    return pl.DataFrame({"user_id": users, "channel_hour_cosine_7_28": cosine,
                         "recent_gate": np.clip(2*(cosine-.5), 0, 1)})


def specialist_features(frame, views, future, cutoff, end):
    """Reuse phase-eight features; explicitly de-log accumulated counts before rates."""
    history_days = (cutoff-TRAIN_START).total_seconds()/86400
    prediction_days = (end-cutoff).total_seconds()/86400
    air = future.with_columns(
        pl.col("programme_start_utc").dt.convert_time_zone("Europe/Kyiv").dt.hour().cast(pl.Int64).alias("hour"),
        pl.col("programme_start_utc").dt.convert_time_zone("Europe/Kyiv").dt.weekday().cast(pl.Int64).alias("dow"),
        pl.lit(1.).alias("weight"))
    recent = clip_views(views, cutoff-timedelta(days=7), cutoff)
    long = clip_views(views, TRAIN_START, cutoff)
    frame = frame.with_columns(
        pl.Series("compatibility_dow_hour_7", compatibility(frame, recent, air, ["dow", "hour"])),
        pl.Series("compatibility_channel_long", compatibility(frame, long, air, ["channel_id"])),
        pl.Series("compatibility_channel_hour_long", compatibility(frame, long, air, ["channel_id", "hour"])))
    base = ["rank_global", "rank_frequency", "rank_recency", "rank_ppr", "rank_cbf", "rank_pool",
            "frequency_share", "mean_coverage_all", "max_coverage_all", "observed_days_ago",
            "normalized_score_item_neighbors", "normalized_score_ppr", "normalized_score_frequency",
            "normalized_score_recency", "active_day_fraction", "frequency_per_active_day"]
    recent_cols = ["compatibility_channel_7", "compatibility_hour_7", "compatibility_dow_7",
                   "compatibility_channel_hour_7", "compatibility_dow_hour_7", "duration_mismatch_7",
                   "programme_rate_7", "programme_active_rate_7", "programme_exposure_rate_7",
                   "activity_shift_7", "programme_shift_7", "active_fraction_7"]
    normalized = ["compatibility_channel_7", "compatibility_channel_hour_7", "compatibility_dow_hour_7",
                  "programme_rate_7", "programme_exposure_rate_7"]
    frame = frame.with_columns([(pl.col(c)/pl.col(c).max().over("user_id").clip(1e-12)).alias("user_normalized_"+c) for c in normalized])
    a = base + recent_cols + ["user_normalized_"+c for c in normalized] + ["total_broadcasts_watched", "num_qualifying_broadcasts", "recency_frequency", "global_users"]
    accumulated = ["n20", "n50", "n80", "n300", "meaningful", "strength_seconds",
                   "total_broadcasts_watched", "user_frequency", "global_frequency"]
    expressions = [(pl.col(c).exp()-1).clip(0).truediv(history_days).log1p().alias("history_rate_"+c) for c in accumulated]
    expressions += [(pl.col(c)/prediction_days).log1p().alias("future_rate_"+c) for c in ("score_exposure_10", "score_exposure_80")]
    expressions += [((pl.col("score_recurrence").exp()-1)/prediction_days).log1p().alias("recurrence_per_prediction_day"),
                    ((pl.col("qualified_14").exp()-1)/14).log1p().alias("qualified_rate_14"),
                    ((pl.col("qualified_28").exp()-1)/28).log1p().alias("qualified_rate_28"),
                    ((pl.col("n80").exp()-1)/(pl.col("historical_airings_per_day")*history_days+1)).alias("n80_opportunity_fraction"),
                    ((pl.col("n50").exp()-1)/(pl.col("historical_airings_per_day")*history_days+1)).alias("n50_opportunity_fraction")]
    frame = frame.with_columns(expressions)
    rate_names = ["history_rate_"+c for c in accumulated] + ["future_rate_score_exposure_10", "future_rate_score_exposure_80",
        "recurrence_per_prediction_day", "qualified_rate_14", "qualified_rate_28", "n80_opportunity_fraction", "n50_opportunity_fraction"]
    b = base + rate_names + ["historical_airings_per_day", "future_airings_per_day", "airing_rate_shift",
        "consumption_per_opportunity", "completion_per_opportunity", "programme_rate_14", "programme_rate_28",
        "programme_shift_28", "activity_shift_28", "compatibility_channel_long", "compatibility_channel_hour_long",
        "duration_mismatch_28", "programme_stability", "channel_stability", "n20", "n50", "n80", "n300"]
    for cols in (a, b):
        assert len(cols) == len(set(cols))
    return frame.with_columns([pl.col(c).cast(pl.Float32) for c in set(a+b)]), {"A_recent": a, "B_horizon": b}


def margin_probabilities(frame, margins):
    if len(margins) != frame.height or not np.isfinite(margins).all():
        raise ValueError("Invalid or misaligned margins")
    f = frame.select("user_id").with_columns(pl.Series("m", margins))
    return f.with_columns((pl.col("m")-pl.col("m").max().over("user_id")).exp().alias("p")).with_columns(
        (pl.col("p")/pl.col("p").sum().over("user_id")).alias("p"))["p"].to_numpy()


def fuse_probabilities(frame, recent, horizon, gates=None):
    if len(recent) != frame.height or len(horizon) != frame.height:
        raise ValueError("Candidate alignment mismatch")
    if gates is None:
        weight = .5
    else:
        joined = frame.select("user_id").join(gates.select("user_id", "recent_gate"), on="user_id", how="left", maintain_order="left")
        if joined.height != frame.height or joined["recent_gate"].null_count():
            raise ValueError("Missing or duplicate user gates")
        weight = joined["recent_gate"].to_numpy()
        if not np.isfinite(weight).all() or np.any((weight < 0) | (weight > 1)):
            raise ValueError("Gate outside [0,1]")
    return weight*recent+(1-weight)*horizon
