"""
MAP@5 evaluation module for SWEET.TV recommendation system.

This module provides a standalone, unit-tested implementation of MAP@5
with binary relevance for the validation split.

CRITICAL: Uses fixed evaluation populations (all known users at cutoff + submission users)
and schedule-derived eligible programmes - NO TARGET LEAKAGE.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import polars as pl

logger = logging.getLogger(__name__)


def average_precision_at_k(
    recommended: List[int],
    relevant: Set[int],
    k: int = 5
) -> float:
    """
    Compute Average Precision at k for binary relevance.

    AP@k = sum_{i=1..k} (Precision@i * rel_i) / min(k, |relevant|)
    where rel_i = 1 if item at rank i is relevant AND not previously counted, else 0.

    For users with zero relevant items, AP@k = 0 (contributes to MAP denominator).

    Args:
        recommended: List of recommended item IDs (ordered, length >= k)
        relevant: Set of relevant item IDs
        k: Cutoff rank

    Returns:
        AP@k score in [0, 1]
    """
    # Truncate to top-k
    recommended_k = recommended[:k]

    # Compute precision at each relevant hit (only first occurrence counts)
    hits = 0
    ap_sum = 0.0
    seen_relevant = set()
    for i, item in enumerate(recommended_k, 1):
        if item in relevant and item not in seen_relevant:
            hits += 1
            seen_relevant.add(item)
            ap_sum += hits / i

    # Normalize by min(k, |relevant|)
    # If relevant is empty, denominator = 0, but AP = 0 by convention
    if not relevant:
        return 0.0
    return ap_sum / min(k, len(relevant))


def map_at_k(
    recommendations: Dict[str, List[int]],
    ground_truth: Dict[str, Set[int]],
    k: int = 5,
    evaluation_users: Optional[List[str]] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Compute Mean Average Precision at k over a FIXED evaluation population.

    Every user in evaluation_users contributes exactly one AP@k value.
    Users with no relevant items contribute AP = 0.
    Users missing from recommendations contribute AP = 0 (not skipped).

    Args:
        recommendations: Dict of user_id -> list of recommended programme IDs (length >= k)
        ground_truth: Dict of user_id -> set of relevant programme IDs
        k: Cutoff rank
        evaluation_users: List of user IDs to evaluate (the FIXED evaluation population)

    Returns:
        Tuple of (MAP@k, per-user AP@k dict)
    """
    if evaluation_users is None:
        raise ValueError("evaluation_users must be provided for fixed-population MAP@k")

    ap_scores = {}
    total_ap = 0.0

    for user_id in evaluation_users:
        # Every user contributes exactly one AP value
        recommended = recommendations.get(user_id, [])
        relevant = ground_truth.get(user_id, set())

        ap = average_precision_at_k(recommended, relevant, k)
        ap_scores[user_id] = ap
        total_ap += ap

    map_score = total_ap / len(evaluation_users) if evaluation_users else 0.0
    return map_score, ap_scores


def hit_rate_at_k(
    recommendations: Dict[str, List[int]],
    ground_truth: Dict[str, Set[int]],
    k: int = 5,
    evaluation_users: Optional[List[str]] = None
) -> float:
    """
    Compute Hit Rate at k over a FIXED evaluation population.

    Every user in evaluation_users contributes exactly one hit indicator.

    Args:
        recommendations: Dict of user_id -> list of recommended programme IDs
        ground_truth: Dict of user_id -> set of relevant programme IDs
        k: Cutoff rank
        evaluation_users: List of user IDs to evaluate (the FIXED evaluation population)

    Returns:
        Hit Rate at k in [0, 1]
    """
    if evaluation_users is None:
        raise ValueError("evaluation_users must be provided for fixed-population HitRate@k")

    hits = 0

    for user_id in evaluation_users:
        recommended = set(recommendations.get(user_id, [])[:k])
        relevant = ground_truth.get(user_id, set())

        if recommended & relevant:
            hits += 1

    return hits / len(evaluation_users) if evaluation_users else 0.0


def candidate_coverage(
    recommendations: Dict[str, List[int]],
    eligible_programmes: Set[int],
    k: int = 5
) -> float:
    """
    Compute candidate coverage = fraction of eligible programmes ever recommended.

    Args:
        recommendations: Dict of user_id -> list of recommended programme IDs
        eligible_programmes: Set of all eligible programme IDs
        k: Cutoff rank

    Returns:
        Coverage in [0, 1]
    """
    if not eligible_programmes:
        return 0.0

    recommended_items = set()
    for recs in recommendations.values():
        recommended_items.update(recs[:k])

    return len(recommended_items & eligible_programmes) / len(eligible_programmes)


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

    # Ensure all evaluation users have an entry (empty set if no relevant)
    if evaluation_users:
        for u in evaluation_users:
            if u not in ground_truth:
                ground_truth[u] = set()

    return ground_truth


def load_submission_users(submission_path: str) -> Set[str]:
    """Load user IDs from submission template."""
    df = pl.read_csv(submission_path, schema_overrides={"user_id": pl.Utf8})
    return set(df["user_id"].to_list())


def evaluate_model_fixed(
    model_name: str,
    recommendations: Dict[str, List[int]],
    val_interactions: pl.DataFrame,
    train_interactions: pl.DataFrame,
    submission_users: Optional[Set[str]] = None,
    eligible_programmes: Optional[Set[int]] = None,
    k: int = 5
) -> Dict:
    """
    Run full evaluation for a model using FIXED evaluation populations.

    This function:
    - Uses ALL known users at training cutoff as Population A
    - Uses ALL submission template users as Population B
    - Computes MAP@5 over the FULL populations (denominator = full population)
    - Users with no relevant items contribute AP=0
    - Users missing from recommendations contribute AP=0
    - Uses schedule-derived eligible programmes (no target leakage)

    Args:
        model_name: Name of the model
        recommendations: Dict user_id -> top-k programme IDs
        val_interactions: Validation interactions DataFrame
        train_interactions: Training interactions DataFrame
        submission_users: Set of user IDs from submission template
        eligible_programmes: Set of eligible programme IDs (schedule-derived)
        k: Cutoff rank

    Returns:
        Dict with evaluation metrics
    """
    # Load submission users if not provided
    if submission_users is None:
        submission_users = load_submission_users("data/submission.csv")

    # Get schedule-derived eligible programmes if not provided
    if eligible_programmes is None:
        from .baselines import get_validation_eligible_programmes
        eligible_programmes = get_validation_eligible_programmes()

    # Get FIXED evaluation populations
    all_users, sub_users, relevant_counts = get_evaluation_users(
        val_interactions, train_interactions, submission_users
    )

    # Build ground truth for all evaluation users
    ground_truth = build_ground_truth(val_interactions, positive_only=True, evaluation_users=all_users)

    # Ensure recommendations exist for all evaluation users
    # Missing users get empty list (AP=0, Hit=0)
    all_recs = {u: recommendations.get(u, []) for u in all_users}
    sub_recs = {u: recommendations.get(u, []) for u in sub_users}

    # MAP@5 over FULL populations
    map_all, ap_all = map_at_k(all_recs, ground_truth, k, all_users)
    map_sub, ap_sub = map_at_k(sub_recs, ground_truth, k, sub_users)

    # HitRate@5 over FULL populations
    hr_all = hit_rate_at_k(all_recs, ground_truth, k, all_users)
    hr_sub = hit_rate_at_k(sub_recs, ground_truth, k, sub_users)

    # Candidate coverage
    coverage = candidate_coverage(recommendations, eligible_programmes, k)

    # Compute fallback percentage from actual recommendation generation
    fallback_count = 0
    total_users_with_recs = len([u for u in recommendations if recommendations[u]])
    # Note: This requires access to recommender internals; baseline code fills this

    stats = {
        "model": model_name,
        "map_at_k_all": map_all,
        "map_at_k_submission": map_sub,
        "hit_rate_at_k_all": hr_all,
        "hit_rate_at_k_submission": hr_sub,
        "candidate_coverage": coverage,
        "num_evaluation_users_all": len(all_users),
        "num_evaluation_users_submission": len(sub_users),
        "num_users_with_relevant_all": sum(1 for u in all_users if relevant_counts.get(u, 0) > 0),
        "num_users_with_relevant_submission": sum(1 for u in sub_users if relevant_counts.get(u, 0) > 0),
        "num_users_zero_relevant_all": sum(1 for u in all_users if relevant_counts.get(u, 0) == 0),
        "num_users_zero_relevant_submission": sum(1 for u in sub_users if relevant_counts.get(u, 0) == 0),
        "distinct_recommended_programmes": len(set().union(*[set(recs) for recs in recommendations.values() if recs])),
        "eligible_programmes_count": len(eligible_programmes),
        "fallback_percentage": 0.0,  # Filled by baselines
        "runtime_seconds": 0.0,  # Filled by baselines
    }

    logger.info(f"{model_name}: MAP@5(all)={map_all:.6f}, MAP@5(sub)={map_sub:.6f}, "
                f"HitRate@5(all)={hr_all:.6f}, Coverage={coverage:.4f}, "
                f"Users(all)={len(all_users)}, Users(sub)={len(sub_users)}, "
                f"ZeroRelevant(all)={stats['num_users_zero_relevant_all']}")

    return stats


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
    for u in all_known_users:
        if u not in relevant_counts:
            relevant_counts[u] = 0

    return all_known_users, sub_users, relevant_counts


# Synthetic unit tests with known AP@5 values
def _test_average_precision():
    """Test AP@5 with known values."""
    # Perfect recommendation
    rec = [1, 2, 3, 4, 5]
    rel = {1, 2, 3, 4, 5}
    assert abs(average_precision_at_k(rec, rel, 5) - 1.0) < 1e-9

    # No hits
    rec = [6, 7, 8, 9, 10]
    assert abs(average_precision_at_k(rec, rel, 5) - 0.0) < 1e-9

    # Partial hits at ranks 1, 3, 5
    rec = [1, 6, 2, 7, 3]
    # Precision@1 = 1/1, Precision@3 = 2/3, Precision@5 = 3/5
    # AP = (1 + 2/3 + 3/5) / 5
    expected = (1.0 + 2.0/3.0 + 3.0/5.0) / 5.0
    assert abs(average_precision_at_k(rec, rel, 5) - expected) < 1e-9

    # Relevant items < k
    rec = [1, 2, 3, 4, 5]
    rel_small = {1, 2}
    # Hits at 1, 2 -> Precision@1=1, Precision@2=1
    # AP = (1 + 1) / 2 = 1.0
    assert abs(average_precision_at_k(rec, rel_small, 5) - 1.0) < 1e-9

    # User with no relevant items -> AP=0
    assert abs(average_precision_at_k([1, 2, 3, 4, 5], set(), 5) - 0.0) < 1e-9

    # Duplicate recommendations (should not count twice)
    rec = [1, 1, 2, 3, 4]
    # First hit at rank 1 (1), second 1 is duplicate, then 2 at rank 3, 3 at rank 4, 4 at rank 5
    # Precision@1=1, Precision@3=2/3, Precision@4=3/4, Precision@5=4/5
    # AP = (1 + 2/3 + 3/4 + 4/5) / 4
    expected = (1.0 + 2.0/3.0 + 3.0/4.0 + 4.0/5.0) / 4.0
    assert abs(average_precision_at_k(rec, {1, 2, 3, 4}, 5) - expected) < 1e-9

    print("All synthetic AP@5 tests passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _test_average_precision()