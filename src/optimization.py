"""Shared phase-five windows, scoring and reproducible evidence helpers."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import json
import time

import polars as pl

from .constants import PipelineConfig, TRAIN_START, TRAIN_END, VAL_END
from .interaction_pipeline import DataIngestion, InteractionBuilder, TemporalSplitter
from .evaluation import build_ground_truth, map_at_k, hit_rate_at_k, candidate_coverage
from .submission import template_users, validate_predictions

PROCESSED = Path("data/processed")
EARLY_CUTOFF = datetime(2020, 4, 13, tzinfo=ZoneInfo("Europe/Kyiv")).astimezone(ZoneInfo("UTC"))


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def msk_now():
    return datetime.now(ZoneInfo("Europe/Moscow")).isoformat()


def historical_viewing():
    # Reuse normalized caches, union duplicated boundary sessions by downstream interval union.
    cols = ["user_id", "channel_id", "view_start_utc", "view_end_utc", "duraton_seconds"]
    return pl.concat([pl.scan_parquet(PROCESSED / f"{p}_viewing_clipped.parquet").select(cols)
                      for p in ("train", "val")])


def historical_schedule():
    ingestion = DataIngestion(PipelineConfig())
    return pl.concat([ingestion.load_schedule(p) for p in ("11-20", "21-30")])


def window_interactions(name, start, end):
    """Build a new window cache without changing any earlier experiment files.

    Airings crossing a window edge retain their full duration denominator; only
    viewing inside [start,end) is used. Full-history reconstruction unions the
    two clipped pieces so the May boundary cannot double-count a broadcast.
    """
    path = PROCESSED / f"phase05_{name}.parquet"
    if not path.exists():
        config = PipelineConfig()
        views = historical_viewing().filter(
            (pl.col("view_start_utc") < end) & (pl.col("view_end_utc") > start)
        ).with_columns(
            pl.max_horizontal("view_start_utc", pl.lit(start)).alias("view_start_utc"),
            pl.min_horizontal("view_end_utc", pl.lit(end)).alias("view_end_utc"),
        ).with_columns((pl.col("view_end_utc") - pl.col("view_start_utc"))
                       .dt.total_seconds().alias("duraton_seconds"))
        frame = InteractionBuilder(config).aggregate_user_programme_interactions(views, historical_schedule())
        # Broadcast end can exceed cutoff; it is not future viewing. Clamp the
        # recency proxy to the observation horizon, without shortening duration.
        frame = frame.with_columns(
            pl.max_horizontal("first_interaction_utc", pl.lit(start)).alias("first_interaction_utc"),
            pl.min_horizontal("last_interaction_utc", pl.lit(end)).alias("last_interaction_utc"),
        ).sort(["user_id", "tv_show_id"])
        frame.write_parquet(path)
    return pl.read_parquet(path)


def raw_users(start, end):
    return historical_viewing().filter((pl.col("view_start_utc") < end) &
                                      (pl.col("view_end_utc") > start))\
        .select("user_id").unique().sort("user_id").collect()["user_id"].to_list()


def populations():
    return raw_users(TRAIN_START, TRAIN_END), template_users()


def evaluate(name, recs, targets, all_users, sub_users, eligible, runtime):
    users = list(dict.fromkeys(all_users + sub_users))
    validate_predictions({u: recs[u] for u in users}, users, eligible)
    gt = build_ground_truth(targets)
    result = {"model": name, "runtime_seconds": runtime,
              "eligible_programmes_count": len(eligible),
              "candidate_coverage": candidate_coverage(recs, eligible),
              "distinct_recommended_programmes": len({i for v in recs.values() for i in v})}
    for suffix, cohort in (("all", all_users), ("submission", sub_users)):
        result[f"map_at_k_{suffix}"] = map_at_k(recs, gt, 5, cohort)[0]
        result[f"hit_rate_at_k_{suffix}"] = hit_rate_at_k(recs, gt, 5, cohort)
        result[f"num_evaluation_users_{suffix}"] = len(cohort)
        result[f"num_users_with_relevant_{suffix}"] = sum(bool(gt.get(u)) for u in cohort)
        result[f"num_users_zero_relevant_{suffix}"] = sum(not gt.get(u) for u in cohort)
    return result


def model_rankings(model, users, depth=100):
    # Existing baselines have complete fallback lists; retain full valid prefixes.
    return model.recommend_batch(users, depth)


def rrf(rankings, weights, users, eligible, k=5, constant=10):
    """Weighted reciprocal-rank fusion, with numeric ID tie-breaking."""
    if any(w < 0 for w in weights) or not any(weights):
        raise ValueError("Fusion weights must be nonnegative and not all zero")
    output = {}
    for u in users:
        scores = {}
        for ranking, weight in zip(rankings, weights):
            if not weight:
                continue
            seen = set()
            for rank, item in enumerate(ranking.get(u, []), 1):
                if item in eligible and item != 0 and item not in seen:
                    scores[item] = scores.get(item, 0.0) + weight / (constant + rank)
                    seen.add(item)
        ordered = sorted(scores, key=lambda i: (-scores[i], i))
        if len(ordered) < k:
            ordered += sorted(eligible - {0} - set(ordered))[:k-len(ordered)]
        output[u] = ordered[:k]
    return output
