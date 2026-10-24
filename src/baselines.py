"""
Baseline recommenders for SWEET.TV recommendation system.

Implements three efficient baseline recommenders:
A. Global popularity
B. Personalized historical frequency
C. Recency-weighted personalization
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import polars as pl

from .constants import TRAIN_END
from .evaluation import load_submission_users

logger = logging.getLogger(__name__)


class GlobalPopularityRecommender:
    """
    Baseline A: Global popularity.

    Ranks eligible programmes by historical popularity in training period.
    """

    def __init__(self, score_by: str = "user_count"):
        """
        Args:
            score_by: "user_count" (unique users) or "broadcast_count" (total interactions)
        """
        self.score_by = score_by
        self.programme_scores: Dict[int, float] = {}
        self.global_ranking: List[int] = []

    @staticmethod
    def _to_lazy(df) -> pl.LazyFrame:
        """Convert DataFrame or LazyFrame to LazyFrame."""
        if isinstance(df, pl.LazyFrame):
            return df
        return df.lazy()

    def fit(
        self,
        train_interactions,
        eligible_programmes: Set[int]
    ) -> "GlobalPopularityRecommender":
        """
        Fit on training interactions.

        Args:
            train_interactions: Training user-programme interactions (DataFrame or LazyFrame)
            eligible_programmes: Set of programme IDs available in validation period
        """
        train_lf = self._to_lazy(train_interactions)
        # Filter to positive interactions only
        positive = train_lf.filter(pl.col("is_relevant"))

        if self.score_by == "user_count":
            scores = positive.group_by("tv_show_id").agg(
                pl.col("user_id").n_unique().alias("score")
            )
        elif self.score_by == "broadcast_count":
            scores = positive.group_by("tv_show_id").agg(
                pl.col("total_broadcasts_watched").sum().alias("score")
            )
        elif self.score_by == "watched_seconds":
            scores = positive.group_by("tv_show_id").agg(
                pl.col("total_watched_seconds").sum().alias("score")
            )
        else:
            raise ValueError(f"Unknown score_by: {self.score_by}")

        # Filter to eligible programmes
        scores = scores.filter(pl.col("tv_show_id").is_in(list(eligible_programmes)))

        # Cast score to Float64 for consistency
        scores = scores.with_columns(pl.col("score").cast(pl.Float64))

        # Ensure ALL eligible programmes are in the ranking (score 0 for unseen)
        seen_programmes = set(scores.collect()["tv_show_id"].to_list())
        for prog in eligible_programmes:
            if prog not in seen_programmes:
                # Add with score 0
                scores = pl.concat([scores, pl.LazyFrame({"tv_show_id": [prog], "score": [0.0]})], how="vertical")

        # Sort by score descending, then by tv_show_id for determinism
        scores = scores.sort(["score", "tv_show_id"], descending=[True, False])

        scores_df = scores.collect()
        self.programme_scores = dict(zip(scores_df["tv_show_id"].to_list(), scores_df["score"].to_list()))
        self.global_ranking = scores_df["tv_show_id"].to_list()

        logger.info(f"GlobalPopularityRecommender fitted: {len(self.global_ranking)} eligible programmes")
        return self

    def recommend(self, user_id: str, k: int = 5) -> List[int]:
        """Return top-k programmes for user (same for all users)."""
        return self.global_ranking[:k]

    def recommend_batch(self, user_ids: List[str], k: int = 5) -> Dict[str, List[int]]:
        """Return recommendations for multiple users."""
        return {uid: self.recommend(uid, k) for uid in user_ids}


class PersonalizedFrequencyRecommender:
    """
    Baseline B: Personalized historical frequency.

    Ranks programmes by user's training interaction frequency.
    Falls back to global popularity for cold-start users.
    """

    def __init__(self, score_by: str = "interaction_count", fallback: Optional["GlobalPopularityRecommender"] = None):
        """
        Args:
            score_by: "interaction_count" (total_broadcasts_watched), "watched_seconds", or "frequency"
            fallback: Fallback recommender for cold-start users
        """
        self.score_by = score_by
        self.fallback = fallback
        self.user_rankings: Dict[str, List[int]] = {}
        self.global_ranking: List[int] = []

    @staticmethod
    def _to_lazy(df) -> pl.LazyFrame:
        """Convert DataFrame or LazyFrame to LazyFrame."""
        if isinstance(df, pl.LazyFrame):
            return df
        return df.lazy()

    def fit(
        self,
        train_interactions,
        eligible_programmes: Set[int]
    ) -> "PersonalizedFrequencyRecommender":
        """
        Fit on training interactions.

        Args:
            train_interactions: Training user-programme interactions (DataFrame or LazyFrame)
            eligible_programmes: Set of programme IDs available in validation period
        """
        train_lf = self._to_lazy(train_interactions)
        # Filter to positive interactions
        positive = train_lf.filter(pl.col("is_relevant"))

        if self.score_by == "interaction_count":
            score_col = "total_broadcasts_watched"
        elif self.score_by == "watched_seconds":
            score_col = "total_watched_seconds"
        elif self.score_by == "frequency":
            score_col = "total_broadcasts_watched"
        else:
            raise ValueError(f"Unknown score_by: {self.score_by}")

        # Get user-specific rankings
        user_scores = positive.group_by(["user_id", "tv_show_id"]).agg(
            pl.col(score_col).sum().alias("score")
        ).filter(pl.col("tv_show_id").is_in(list(eligible_programmes)))

        # Sort per user
        user_scores = user_scores.sort(["user_id", "score", "tv_show_id"], descending=[False, True, False])

        # Build per-user rankings
        user_scores_df = user_scores.collect()
        self.user_rankings = {}
        for user_id, group in user_scores_df.partition_by("user_id", as_dict=True).items():
            # user_id is a tuple like ('u1',), extract the string
            uid = user_id[0] if isinstance(user_id, tuple) else user_id
            ranking = group["tv_show_id"].to_list()
            self.user_rankings[uid] = ranking

        # Build global fallback
        if self.fallback is None:
            global_scores = positive.group_by("tv_show_id").agg(
                pl.col("user_id").n_unique().alias("score")
            ).filter(pl.col("tv_show_id").is_in(list(eligible_programmes)))
            
            # Cast score to Float64 for consistency
            global_scores = global_scores.with_columns(pl.col("score").cast(pl.Float64))
            
            # Ensure ALL eligible programmes are in the ranking
            seen = set(global_scores.collect()["tv_show_id"].to_list())
            for prog in eligible_programmes:
                if prog not in seen:
                    global_scores = pl.concat([global_scores, pl.LazyFrame({"tv_show_id": [prog], "score": [0.0]})], how="vertical")
            
            global_scores = global_scores.sort(["score", "tv_show_id"], descending=[True, False])
            self.global_ranking = global_scores.collect()["tv_show_id"].to_list()
        else:
            self.global_ranking = self.fallback.global_ranking

        logger.info(f"PersonalizedFrequencyRecommender fitted: {len(self.user_rankings)} users with history")
        return self

    def recommend(self, user_id: str, k: int = 5) -> List[int]:
        """Return top-k programmes for user with fallback."""
        if user_id in self.user_rankings:
            personal = self.user_rankings[user_id]
            if len(personal) >= k:
                return personal[:k]
            # Need fallback for remaining slots
            remaining = k - len(personal)
            personal_set = set(personal)
            fallback = [p for p in self.global_ranking if p not in personal_set][:remaining]
            return personal + fallback
        else:
            # Cold start - use global ranking
            return self.global_ranking[:k]

    def recommend_batch(self, user_ids: List[str], k: int = 5) -> Dict[str, List[int]]:
        """Return recommendations for multiple users."""
        return {uid: self.recommend(uid, k) for uid in user_ids}


class RecencyWeightedRecommender:
    """
    Baseline C: Recency-weighted personalization.

    Applies exponential time decay to training interactions.
    """

    def __init__(
        self,
        half_life_days: float = 14.0,
        score_by: str = "interaction_count",
        fallback: Optional[GlobalPopularityRecommender] = None,
        cutoff: datetime = TRAIN_END,
    ):
        """
        Args:
            half_life_days: Half-life for exponential decay in days
            score_by: "interaction_count" or "watched_seconds"
            fallback: Fallback recommender for cold-start users
        """
        self.half_life_days = half_life_days
        self.cutoff = cutoff
        self.score_by = score_by
        self.fallback = fallback
        self.user_rankings: Dict[str, List[int]] = {}
        self.global_ranking: List[int] = []
        self.decay_lambda = 1.0  # Will be set in fit

    @staticmethod
    def _to_lazy(df) -> pl.LazyFrame:
        """Convert DataFrame or LazyFrame to LazyFrame."""
        if isinstance(df, pl.LazyFrame):
            return df
        return df.lazy()

    def _compute_decay(self, days_ago: float) -> float:
        """Compute decay weight for an interaction `days_ago` days before cutoff."""
        import math
        if self.half_life_days <= 0:
            return 1.0
        # Exponential decay: exp(-lambda * t) where lambda = ln(2) / half_life
        return math.exp(-math.log(2) * days_ago / self.half_life_days)

    def fit(
        self,
        train_interactions,
        eligible_programmes: Set[int]
    ) -> "RecencyWeightedRecommender":
        """
        Fit on training interactions with recency weighting.

        Args:
            train_interactions: Training user-programme interactions (DataFrame or LazyFrame)
            eligible_programmes: Set of programme IDs available in validation period
        """
        import math

        train_lf = self._to_lazy(train_interactions)
        # Filter to positive interactions
        positive = train_lf.filter(pl.col("is_relevant")).collect()

        if positive.height == 0:
            self.user_rankings = {}
            self.global_ranking = []
            return self

        # Use TRAIN_END as the cutoff (already UTC)
        cutoff = self.cutoff

        # Compute days since last interaction for each user-programme pair
        # We need to use the original viewing timestamps, not the aggregated last_interaction_utc
        # For now, use the aggregated last_interaction_utc as a proxy
        positive = positive.with_columns([
            ((pl.lit(cutoff) - pl.col("last_interaction_utc")).dt.total_seconds() / 86400.0).alias("days_ago")
        ]).with_columns([
            pl.col("days_ago").map_elements(self._compute_decay, return_dtype=pl.Float64).alias("decay_weight")
        ])

        if self.score_by == "interaction_count":
            base_score = "total_broadcasts_watched"
        elif self.score_by == "watched_seconds":
            base_score = "total_watched_seconds"
        else:
            raise ValueError(f"Unknown score_by: {self.score_by}")

        # Weighted score
        positive = positive.with_columns([
            (pl.col(base_score) * pl.col("decay_weight")).alias("weighted_score")
        ])

        # Aggregate per user-programme
        user_scores = positive.group_by(["user_id", "tv_show_id"]).agg(
            pl.col("weighted_score").sum().alias("score")
        ).filter(pl.col("tv_show_id").is_in(list(eligible_programmes)))

        # Sort per user
        user_scores = user_scores.sort(["user_id", "score", "tv_show_id"], descending=[False, True, False])

        # Build per-user rankings
        self.user_rankings = {}
        for user_id, group in user_scores.partition_by("user_id", as_dict=True).items():
            # user_id is a tuple like ('u1',), extract the string
            uid = user_id[0] if isinstance(user_id, tuple) else user_id
            ranking = group["tv_show_id"].to_list()
            self.user_rankings[uid] = ranking

        # Build global fallback
        if self.fallback is None:
            global_scores = positive.group_by("tv_show_id").agg(
                pl.col("weighted_score").sum().alias("score")
            ).filter(pl.col("tv_show_id").is_in(sorted(eligible_programmes - {0})))
            
            # Cast score to Float64 for consistency
            global_scores = global_scores.with_columns(pl.col("score").cast(pl.Float64))
            
            # Ensure ALL eligible programmes are in the ranking
            seen = set(global_scores["tv_show_id"].to_list())
            for prog in eligible_programmes:
                if prog not in seen:
                    global_scores = pl.concat([global_scores, pl.DataFrame({"tv_show_id": [prog], "score": [0.0]})], how="vertical")
            
            global_scores = global_scores.sort(["score", "tv_show_id"], descending=[True, False])
            self.global_ranking = global_scores["tv_show_id"].to_list()
        else:
            self.global_ranking = self.fallback.global_ranking

        logger.info(f"RecencyWeightedRecommender (half_life={self.half_life_days}d) fitted: "
                    f"{len(self.user_rankings)} users with history")
        return self

    def recommend(self, user_id: str, k: int = 5) -> List[int]:
        """Return top-k programmes for user with fallback."""
        if user_id in self.user_rankings:
            personal = self.user_rankings[user_id]
            if len(personal) >= k:
                return personal[:k]
            remaining = k - len(personal)
            personal_set = set(personal)
            fallback = [p for p in self.global_ranking if p not in personal_set][:remaining]
            return personal + fallback
        else:
            return self.global_ranking[:k]

    def recommend_batch(self, user_ids: List[str], k: int = 5) -> Dict[str, List[int]]:
        """Return recommendations for multiple users."""
        return {uid: self.recommend(uid, k) for uid in user_ids}


import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import polars as pl

from .constants import TRAIN_END, VAL_START, VAL_END
from .evaluation import load_submission_users, evaluate_model_fixed, get_evaluation_users

logger = logging.getLogger(__name__)


def get_schedule_eligible_programmes(
    schedule_lf: pl.LazyFrame,
    window_start: datetime,
    window_end: datetime
) -> Set[int]:
    """
    Get eligible programmes from broadcast schedule for a given time window.
    
    Uses ONLY schedule information - no viewing interactions.
    """
    # Filter schedule to programmes airing within the validation window
    # Programme airs if: programme_start < window_end AND programme_end > window_start
    eligible = schedule_lf.filter(
        (pl.col("programme_start_utc") < pl.lit(window_end)) &
        (pl.col("programme_end_utc") > pl.lit(window_start)) &
        (pl.col("tv_show_id") != 0)
    ).select("tv_show_id").unique().collect()
    
    return set(eligible["tv_show_id"].to_list())


def get_validation_eligible_programmes() -> Set[int]:
    """
    Get eligible programmes for validation period from schedule.
    Reads export_arh_21-30-final.csv and applies UTC validation window.
    """
    from .interaction_pipeline import DataIngestion, PipelineConfig
    from .constants import VAL_START, VAL_END
    
    config = PipelineConfig()
    ingestion = DataIngestion(config)
    schedule_lf = ingestion.load_schedule("21-30")
    
    return get_schedule_eligible_programmes(schedule_lf, VAL_START, VAL_END)


def get_test_eligible_programmes() -> Set[int]:
    """
    Get eligible programmes for test/final period from schedule.
    Reads export_arh_31-42-final.csv and applies UTC test window.
    """
    from .interaction_pipeline import DataIngestion, PipelineConfig
    from .constants import TEST_START, TEST_END
    
    config = PipelineConfig()
    ingestion = DataIngestion(config)
    schedule_lf = ingestion.load_schedule("31-42")
    
    return get_schedule_eligible_programmes(schedule_lf, TEST_START, TEST_END)


def get_evaluation_users(
    val_interactions: pl.DataFrame,
    train_interactions: pl.DataFrame,
    submission_users: Optional[Set[str]] = None
) -> Tuple[List[str], List[str], Dict[str, int]]:
    """
    Get fixed evaluation populations.

    Returns:
        Tuple of (all_known_users, submission_users, relevant_counts_dict)
        where relevant_counts_dict maps user_id -> count of relevant validation programmes
    """
    # Population A: All users with any training interaction (known at cutoff)
    all_known_users = train_interactions.select("user_id").unique().to_series().to_list()
    
    # Population B: ALL submission template users (exactly 1260), NOT intersected with training users
    if submission_users:
        sub_users = list(submission_users)
    else:
        sub_users = []
    
    # Build ground truth for relevant validation programmes per user
    val_positive = val_interactions.filter(pl.col("is_relevant"))
    relevant_counts = {}
    for user_id, count in val_positive.group_by("user_id").len().iter_rows():
        relevant_counts[user_id] = count
    
    # Ensure all known users have an entry (0 if no relevant)
    for u in all_known_users:
        if u not in relevant_counts:
            relevant_counts[u] = 0
    
    # Also ensure submission users have entries (0 if no relevant)
    for u in sub_users:
        if u not in relevant_counts:
            relevant_counts[u] = 0
    
    return all_known_users, sub_users, relevant_counts


def build_ground_truth(
    val_interactions: pl.DataFrame,
    positive_only: bool = True,
    evaluation_users: Optional[List[str]] = None
) -> Dict[str, Set[int]]:
    """
    Build ground truth dict from validation interactions.
    
    Args:
        val_interactions: DataFrame with columns user_id, tv_show_id, is_relevant
        positive_only: If True, only include is_relevant=True programmes
        evaluation_users: If provided, ensure all these users are in output (with empty set if no relevant)
    
    Returns:
        Dict of user_id -> set of relevant tv_show_ids
    """
    if positive_only:
        df = val_interactions.filter(pl.col("is_relevant"))
    else:
        df = val_interactions

    ground_truth = {}
    for user_id, tv_show_id in df.select(["user_id", "tv_show_id"]).iter_rows():
        if user_id not in ground_truth:
            ground_truth[user_id] = set()
        ground_truth[user_id].add(tv_show_id)

    # Ensure all evaluation users have an entry
    if evaluation_users:
        for u in evaluation_users:
            if u not in ground_truth:
                ground_truth[u] = set()

    return ground_truth


def run_baseline_benchmark(
    train_interactions: pl.DataFrame,
    val_interactions: pl.DataFrame,
    submission_path: str,
    eligible_programmes: Optional[Set[int]] = None,
    half_lives: Tuple[float, ...] = (7.0, 14.0, 30.0)
) -> Dict:
    """
    Run complete baseline benchmark with CORRECTED evaluation (no target leakage, fixed populations).

    Returns:
        Dict with evaluation results for all baselines
    """
    # Get schedule-derived eligible programmes if not provided
    if eligible_programmes is None:
        eligible_programmes = get_validation_eligible_programmes()

    submission_users = load_submission_users("data/submission.csv")

    logger.info(f"Schedule-derived eligible programmes: {len(eligible_programmes)}")
    logger.info(f"Training users: {train_interactions.select('user_id').n_unique()}")
    logger.info(f"Validation users: {val_interactions.select('user_id').n_unique()}")

    # Get FIXED evaluation populations ONCE
    submission_users = load_submission_users("data/submission.csv")
    all_users, sub_users, relevant_counts = get_evaluation_users(val_interactions, train_interactions, submission_users)
    
    logger.info(f"Population A (all known users): {len(all_users)}")
    logger.info(f"Population B (submission users): {len(sub_users)}")

    results = {}

    # Baseline A: Global Popularity
    logger.info("Running Baseline A: Global Popularity")
    for score_by in ["user_count", "broadcast_count", "watched_seconds"]:
        start = time.perf_counter()
        rec = GlobalPopularityRecommender(score_by=score_by)
        rec.fit(train_interactions, eligible_programmes)
        
        recs = rec.recommend_batch(all_users, 5)
        fit_time = time.perf_counter() - start
        
        stats = evaluate_model_fixed(
            f"GlobalPop_{score_by}", recs, val_interactions, train_interactions,
            submission_users=submission_users, eligible_programmes=eligible_programmes, k=5
        )
        stats["runtime_seconds"] = fit_time
        stats["fallback_percentage"] = 0.0  # Global pop doesn't use fallback
        results[f"GlobalPop_{score_by}"] = stats

    # Baseline B: Personalized Frequency
    logger.info("Running Baseline B: Personalized Frequency")
    for score_by in ["interaction_count", "watched_seconds"]:
        start = time.perf_counter()
        rec = PersonalizedFrequencyRecommender(score_by=score_by)
        rec.fit(train_interactions, eligible_programmes)
        
        recs = rec.recommend_batch(all_users, 5)
        fit_time = time.perf_counter() - start
        
        # Calculate fallback percentage
        fallback_count = sum(1 for u in rec.user_rankings if len(rec.user_rankings[u]) < 5)
        fallback_pct = fallback_count / len(all_users) * 100 if all_users else 0
        
        stats = evaluate_model_fixed(
            f"PersonalizedFreq_{score_by}", recs, val_interactions, train_interactions,
            submission_users=submission_users, eligible_programmes=eligible_programmes, k=5
        )
        stats["runtime_seconds"] = time.perf_counter() - start
        stats["fallback_percentage"] = fallback_pct
        results[f"PersonalizedFreq_{score_by}"] = stats

    # Baseline C: Recency-Weighted
    logger.info("Running Baseline C: Recency-Weighted")
    for hl in half_lives:
        for score_by in ["interaction_count", "watched_seconds"]:
            start = time.perf_counter()
            rec = RecencyWeightedRecommender(half_life_days=hl, score_by=score_by)
            rec.fit(train_interactions, eligible_programmes)
            
            recs = rec.recommend_batch(all_users, 5)
            fit_time = time.perf_counter() - start
            
            # Calculate fallback percentage
            fallback_count = sum(1 for u in rec.user_rankings if len(rec.user_rankings[u]) < 5)
            fallback_pct = fallback_count / len(all_users) * 100 if all_users else 0
            
            stats = evaluate_model_fixed(
                f"Recency_{hl}d_{score_by}", recs, val_interactions, train_interactions,
                submission_users=submission_users, eligible_programmes=eligible_programmes, k=5
            )
            stats["runtime_seconds"] = time.perf_counter() - start
            stats["fallback_percentage"] = fallback_pct
            results[f"Recency_{hl}d_{score_by}"] = stats

    return results


def _get_evaluation_users(val_interactions: pl.DataFrame, submission_users: Set[str]):
    """Get evaluation user populations."""
    all_users = val_interactions.filter(pl.col("is_relevant")).select("user_id").unique().to_series().to_list()
    sub_users = [u for u in all_users if u in submission_users]
    return all_users, sub_users


def _evaluate_recommender(
    name: str,
    recs: Dict[str, List[int]],
    val_interactions: pl.DataFrame,
    all_users: List[str],
    sub_users: List[str],
    eligible_programmes: Set[int],
    k: int = 5
) -> Dict:
    """Evaluate a recommender."""
    from .evaluation import (
        map_at_k, hit_rate_at_k, candidate_coverage, build_ground_truth
    )

    ground_truth = build_ground_truth(val_interactions, positive_only=True)

    # Filter recs to evaluation users
    recs_all = {u: recs[u] for u in all_users if u in recs}
    recs_sub = {u: recs[u] for u in sub_users if u in recs}

    map_all, _ = map_at_k(recs_all, ground_truth, k, all_users)
    map_sub, _ = map_at_k(recs_sub, ground_truth, k, sub_users) if sub_users else (0.0, {})

    hr_all = hit_rate_at_k(recs_all, ground_truth, k, all_users)
    hr_sub = hit_rate_at_k(recs_sub, ground_truth, k, sub_users) if sub_users else 0.0

    coverage = candidate_coverage(recs, eligible_programmes, k)

    # Fallback percentage
    fallback_count = 0
    # This would need access to the recommender internals
    # For now, we'll estimate from user coverage
    fallback_pct = 0.0

    stats = {
        "model": name,
        "map_at_k_all": map_all,
        "map_at_k_submission": map_sub,
        "hit_rate_at_k_all": hr_all,
        "hit_rate_at_k_submission": hr_sub,
        "candidate_coverage": coverage,
        "num_evaluation_users_all": len(all_users),
        "num_evaluation_users_submission": len(sub_users),
        "fallback_percentage": fallback_pct,
        "distinct_recommended_programmes": len(set().union(*[set(v) for v in recs.values()])),
    }

    logger.info(f"{name}: MAP@5(all)={map_all:.6f}, MAP@5(sub)={map_sub:.6f}, "
                f"HitRate@5(all)={hr_all:.6f}, Coverage={coverage:.4f}")

    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with sample data
    from src.interaction_pipeline import PipelineConfig, run_pipeline

    config = PipelineConfig()
    evidence = run_pipeline(config)

    train_interactions = pl.read_parquet(config.processed_dir / "train_interactions.parquet")
    val_interactions = pl.read_parquet(config.processed_dir / "val_interactions.parquet")

    results = run_baseline_benchmark(train_interactions, val_interactions, "data/submission.csv")

    import json
    with open("evidence/phase03_baselines.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
