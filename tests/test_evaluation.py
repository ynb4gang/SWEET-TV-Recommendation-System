#!/usr/bin/env python3
"""
Unit tests for corrected evaluation module (no target leakage, fixed populations).
"""
import pytest
import polars as pl

from src.evaluation import (
    average_precision_at_k,
    map_at_k,
    hit_rate_at_k,
    candidate_coverage,
    build_ground_truth,
    get_evaluation_users,
    evaluate_model_fixed,
)


class TestAveragePrecisionAtK:
    """Test AP@k with known analytical values."""

    def test_perfect_recommendation(self):
        """All top-5 are relevant."""
        rec = [1, 2, 3, 4, 5]
        rel = {1, 2, 3, 4, 5}
        assert average_precision_at_k(rec, rel, 5) == 1.0

    def test_zero_hits(self):
        """No relevant items in recommendations."""
        rec = [6, 7, 8, 9, 10]
        rel = {1, 2, 3, 4, 5}
        assert average_precision_at_k(rec, rel, 5) == 0.0

    def test_partial_hits_at_known_ranks(self):
        """Hits at ranks 1, 3, 5."""
        rec = [1, 6, 2, 7, 3]
        rel = {1, 2, 3, 4, 5}
        # Precision@1 = 1/1, Precision@3 = 2/3, Precision@5 = 3/5
        # AP = (1 + 2/3 + 3/5) / 5
        expected = (1.0 + 2.0/3.0 + 3.0/5.0) / 5.0
        assert abs(average_precision_at_k(rec, rel, 5) - expected) < 1e-9

    def test_fewer_relevant_than_k(self):
        """Only 2 relevant items, both at top."""
        rec = [1, 2, 3, 4, 5]
        rel = {1, 2}
        # Hits at 1, 2 -> Precision@1=1, Precision@2=1
        # AP = (1 + 1) / 2 = 1.0
        assert average_precision_at_k(rec, rel, 5) == 1.0

    def test_empty_relevant(self):
        """User with no relevant items gets AP=0."""
        rec = [1, 2, 3, 4, 5]
        rel = set()
        assert average_precision_at_k(rec, rel, 5) == 0.0

    def test_duplicate_recommendations(self):
        """Duplicate recommendations should not double-count."""
        rec = [1, 1, 2, 3, 4]
        rel = {1, 2, 3, 4}
        # First hit at rank 1 (1), second 1 is duplicate, then 2 at rank 3, 3 at rank 4, 4 at rank 5
        # Precision@1=1, Precision@3=2/3, Precision@4=3/4, Precision@5=4/5
        # AP = (1 + 2/3 + 3/4 + 4/5) / 4
        expected = (1.0 + 2.0/3.0 + 3.0/4.0 + 4.0/5.0) / 4.0
        assert abs(average_precision_at_k(rec, rel, 5) - expected) < 1e-9

    def test_k_less_than_5(self):
        """Test with k=3."""
        rec = [1, 2, 3]
        rel = {1, 2, 3}
        assert average_precision_at_k(rec, rel, 3) == 1.0


class TestMapAtK:
    """Test MAP@k over fixed populations."""

    def test_map_all_users(self):
        """MAP over multiple users."""
        recommendations = {
            "u1": [1, 2, 3, 4, 5],
            "u2": [6, 7, 8, 9, 10],
        }
        ground_truth = {
            "u1": {1, 2, 3},
            "u2": {6, 7},
        }
        evaluation_users = ["u1", "u2"]
        map_score, ap_scores = map_at_k(recommendations, ground_truth, 5, evaluation_users)
        assert abs(ap_scores["u1"] - 1.0) < 1e-9
        assert abs(ap_scores["u2"] - 1.0) < 1e-9
        assert map_score == 1.0

    def test_map_with_zero_relevant(self):
        """Users with no relevant items get AP=0 and are included in mean."""
        recommendations = {
            "u1": [1, 2, 3, 4, 5],
            "u2": [6, 7, 8, 9, 10],
        }
        ground_truth = {
            "u1": {1, 2, 3},
            "u2": set(),
        }
        evaluation_users = ["u1", "u2"]
        map_score, _ = map_at_k(recommendations, ground_truth, 5, evaluation_users)
        # u1: AP=1.0, u2: AP=0.0 -> MAP = 0.5
        assert abs(map_score - 0.5) < 1e-9

    def test_map_missing_user_in_recommendations(self):
        """Missing user in recommendations contributes AP=0."""
        recommendations = {
            "u1": [1, 2, 3, 4, 5],
        }
        ground_truth = {
            "u1": {1, 2, 3},
            "u2": {6, 7},
        }
        evaluation_users = ["u1", "u2"]
        map_score, _ = map_at_k(recommendations, ground_truth, 5, evaluation_users)
        # u1: AP=1.0, u2: AP=0.0 (missing) -> MAP = 0.5
        assert abs(map_score - 0.5) < 1e-9


class TestHitRateAtK:
    """Test Hit Rate at k over fixed populations."""

    def test_hit_rate_all_hit(self):
        """All users have at least one hit."""
        recommendations = {
            "u1": [1, 2, 3, 4, 5],
            "u2": [6, 7, 8, 9, 10],
        }
        ground_truth = {
            "u1": {1, 2, 3},
            "u2": {6, 7},
        }
        evaluation_users = ["u1", "u2"]
        hr = hit_rate_at_k(recommendations, ground_truth, 5, evaluation_users)
        assert hr == 1.0

    def test_hit_rate_partial(self):
        """Some users have hits."""
        recommendations = {
            "u1": [1, 2, 3, 4, 5],
            "u2": [6, 7, 8, 9, 10],
        }
        ground_truth = {
            "u1": {1, 2, 3},
            "u2": {11, 12},
        }
        evaluation_users = ["u1", "u2"]
        hr = hit_rate_at_k(recommendations, ground_truth, 5, evaluation_users)
        assert hr == 0.5

    def test_hit_rate_none(self):
        """No hits."""
        recommendations = {
            "u1": [1, 2, 3, 4, 5],
            "u2": [6, 7, 8, 9, 10],
        }
        ground_truth = {
            "u1": {11, 12},
            "u2": {13, 14},
        }
        evaluation_users = ["u1", "u2"]
        hr = hit_rate_at_k(recommendations, ground_truth, 5, evaluation_users)
        assert hr == 0.0

    def test_hit_rate_zero_relevant_user(self):
        """User with zero relevant items cannot hit."""
        recommendations = {
            "u1": [1, 2, 3, 4, 5],
            "u2": [6, 7, 8, 9, 10],
        }
        ground_truth = {
            "u1": {1, 2, 3},
            "u2": set(),
        }
        evaluation_users = ["u1", "u2"]
        hr = hit_rate_at_k(recommendations, ground_truth, 5, evaluation_users)
        assert hr == 0.5


class TestCandidateCoverage:
    """Test candidate coverage."""

    def test_coverage_all_recommended(self):
        """All eligible programmes recommended."""
        recommendations = {
            "u1": [1, 2, 3],
            "u2": [4, 5, 6],
        }
        eligible = {1, 2, 3, 4, 5, 6}
        cov = candidate_coverage(recommendations, eligible, 3)
        assert cov == 1.0

    def test_coverage_partial(self):
        """Some eligible programmes never recommended."""
        recommendations = {
            "u1": [1, 2, 3],
            "u2": [1, 2, 3],
        }
        eligible = {1, 2, 3, 4, 5, 6}
        cov = candidate_coverage(recommendations, eligible, 3)
        assert cov == 0.5


class TestGroundTruth:
    """Test ground truth construction."""

    def test_positive_only(self):
        """Only positive interactions included."""
        df = pl.DataFrame({
            "user_id": ["u1", "u1", "u2"],
            "tv_show_id": [1, 2, 1],
            "is_relevant": [True, False, True],
        })
        gt = build_ground_truth(df, positive_only=True)
        assert gt == {"u1": {1}, "u2": {1}}

    def test_all_interactions(self):
        """All interactions included when positive_only=False."""
        df = pl.DataFrame({
            "user_id": ["u1", "u1", "u2"],
            "tv_show_id": [1, 2, 1],
            "is_relevant": [True, False, True],
        })
        gt = build_ground_truth(df, positive_only=False)
        assert gt == {"u1": {1, 2}, "u2": {1}}

    def test_evaluation_users_included(self):
        """All evaluation users included even with no relevant items."""
        df = pl.DataFrame({
            "user_id": ["u1", "u1"],
            "tv_show_id": [1, 2],
            "is_relevant": [True, True],
        })
        gt = build_ground_truth(df, positive_only=True, evaluation_users=["u1", "u2", "u3"])
        assert gt == {"u1": {1, 2}, "u2": set(), "u3": set()}


class TestFixedPopulations:
    """Test fixed evaluation population logic."""

    def test_fixed_population_includes_zero_relevant(self):
        """Fixed population includes users with zero relevant items."""
        recommendations = {
            "u1": [1, 2, 3, 4, 5],
            "u2": [6, 7, 8, 9, 10],
            "u3": [11, 12, 13, 14, 15],
        }
        ground_truth = {
            "u1": {1, 2},
            "u2": {6},
            "u3": set(),
        }
        evaluation_users = ["u1", "u2", "u3"]
        
        map_score, _ = map_at_k(recommendations, ground_truth, 5, evaluation_users)
        # u1: AP=1.0, u2: AP=1.0, u3: AP=0.0 -> MAP = 2/3
        assert abs(map_score - 2.0/3.0) < 1e-9

        hr = hit_rate_at_k(recommendations, ground_truth, 5, evaluation_users)
        # u1: hit, u2: hit, u3: no hit -> HR = 2/3
        assert abs(hr - 2.0/3.0) < 1e-9

    def test_missing_recommendations_are_zero(self):
        """Missing recommendations contribute AP=0, Hit=0."""
        recommendations = {
            "u1": [1, 2, 3, 4, 5],
        }
        ground_truth = {
            "u1": {1, 2},
            "u2": {6, 7},
        }
        evaluation_users = ["u1", "u2"]
        
        map_score, _ = map_at_k(recommendations, ground_truth, 5, evaluation_users)
        assert abs(map_score - 0.5) < 1e-9
        
        hr = hit_rate_at_k(recommendations, ground_truth, 5, evaluation_users)
        assert hr == 0.5


class TestDeterministicTieBreaking:
    """Test deterministic tie-breaking in recommendations."""

    def test_duplicate_recommendations_no_double_credit(self):
        """Duplicate correct programme must not generate additional credit."""
        rec = [1, 1, 2, 3, 4]
        rel = {1, 2, 3, 4}
        # First hit at rank 1, then duplicate 1 ignored
        # Hits at ranks 1, 3, 4, 5 -> Precisions: 1/1, 2/3, 3/4, 4/5
        expected = (1.0 + 2.0/3.0 + 3.0/4.0 + 4.0/5.0) / 4.0
        assert abs(average_precision_at_k(rec, rel, 5) - expected) < 1e-9

    def test_recommendation_length_exactly_k(self):
        """Recommendations must have exactly k items for metric computation."""
        rec = [1, 2, 3, 4, 5]
        rel = {1, 2, 3}
        ap = average_precision_at_k(rec, rel, 5)
        # Hits at 1, 2, 3 -> Precisions: 1/1, 2/2, 3/3 -> sum = 3 -> AP = 3/3 = 1.0
        assert ap == 1.0


class TestForbiddenProgramme0:
    """Test that programme ID 0 is never recommended."""
    
    def test_programme_0_not_in_candidates(self):
        """Programme 0 should never be in eligible set."""
        # This is a semantic test - our code filters out tv_show_id != 0
        from src.submission import validate_predictions
        with pytest.raises(ValueError):
            validate_predictions({"u": [0, 1, 2, 3, 4]}, ["u"], {0, 1, 2, 3, 4, 5})


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
