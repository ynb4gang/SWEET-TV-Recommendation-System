#!/usr/bin/env python3
"""
Unit tests for baseline recommenders.
"""
import pytest
import polars as pl
import numpy as np

from src.baselines import (
    GlobalPopularityRecommender,
    PersonalizedFrequencyRecommender,
    RecencyWeightedRecommender,
    get_validation_eligible_programmes,
)
from src.evaluation import load_submission_users


class TestGlobalPopularityRecommender:
    """Tests for global popularity baseline."""

    def test_fit_and_recommend(self):
        """Fit on train data and produce recommendations."""
        # Create synthetic train interactions
        train = pl.DataFrame({
            "user_id": ["u1", "u1", "u2", "u2", "u3"],
            "tv_show_id": [1, 2, 1, 3, 1],
            "is_relevant": [True, True, True, True, True],
            "total_broadcasts_watched": [1, 1, 1, 1, 2],
        })

        eligible = {1, 2, 3, 4, 5}
        rec = GlobalPopularityRecommender(score_by="user_count")
        rec.fit(train, eligible)

        # Programme 1 has 3 unique users, 2 has 1, 3 has 1
        assert rec.global_ranking[0] == 1
        assert set(rec.global_ranking[:3]) == {1, 2, 3}

        # Recommendations are same for all users
        recs = rec.recommend_batch(["u1", "u4"], 3)
        assert recs["u1"] == recs["u4"] == [1, 2, 3]

    def test_filter_eligible(self):
        """Only eligible programmes are recommended."""
        train = pl.DataFrame({
            "user_id": ["u1", "u2"],
            "tv_show_id": [1, 2],
            "is_relevant": [True, True],
        })

        eligible = {2, 3}  # Only 2 and 3 eligible
        rec = GlobalPopularityRecommender(score_by="user_count")
        rec.fit(train, eligible)

        # Programme 2 has score 1, programme 3 has score 0 (unseen)
        assert rec.global_ranking == [2, 3]
        assert rec.recommend("u1", 5) == [2, 3]

    def test_score_by_broadcast_count(self):
        """Score by total broadcast count."""
        train = pl.DataFrame({
            "user_id": ["u1", "u1", "u2"],
            "tv_show_id": [1, 1, 2],
            "is_relevant": [True, True, True],
            "total_broadcasts_watched": [2, 1, 1],  # u1 watched 1 twice, 2 once
        })

        eligible = {1, 2}
        rec = GlobalPopularityRecommender(score_by="broadcast_count")
        rec.fit(train, eligible)

        # Programme 1: 3 broadcasts, Programme 2: 1 broadcast
        assert rec.global_ranking == [1, 2]

    def test_deterministic_sorting(self):
        """Ties broken by programme ID for determinism."""
        train = pl.DataFrame({
            "user_id": ["u1", "u2"],
            "tv_show_id": [2, 1],  # Reverse order
            "is_relevant": [True, True],
        })

        eligible = {1, 2}
        rec = GlobalPopularityRecommender(score_by="user_count")
        rec.fit(train, eligible)

        # Both have score 1, tie broken by tv_show_id
        assert rec.global_ranking == [1, 2]


class TestPersonalizedFrequencyRecommender:
    """Tests for personalized frequency baseline."""

    def test_personalized_ranking(self):
        """User gets personalized ranking."""
        train = pl.DataFrame({
            "user_id": ["u1", "u1", "u1", "u2", "u2"],
            "tv_show_id": [1, 2, 3, 1, 4],
            "is_relevant": [True] * 5,
            "total_broadcasts_watched": [3, 2, 1, 5, 2],
        })

        eligible = {1, 2, 3, 4, 5}
        rec = PersonalizedFrequencyRecommender(score_by="interaction_count")
        rec.fit(train, eligible)

        # u1: 1(3), 2(2), 3(1) -> ranking [1, 2, 3]
        # u2: 1(5), 4(2) -> ranking [1, 4]
        assert rec.user_rankings["u1"] == [1, 2, 3]
        assert rec.user_rankings["u2"] == [1, 4]

        # Recommendations use personal ranking
        assert rec.recommend("u1", 3) == [1, 2, 3]
        assert rec.recommend("u2", 3) == [1, 4, 2]

    def test_fallback_to_global(self):
        """Cold-start users get global ranking."""
        train = pl.DataFrame({
            "user_id": ["u1", "u1"],
            "tv_show_id": [1, 2],
            "is_relevant": [True, True],
            "total_broadcasts_watched": [1, 1],
        })

        eligible = {1, 2, 3}
        rec = PersonalizedFrequencyRecommender(score_by="interaction_count")
        rec.fit(train, eligible)

        # u3 is cold-start
        recs = rec.recommend("u3", 3)
        # Should use global fallback
        assert len(recs) == 3
        assert set(recs) == {1, 2, 3}

    def test_fallback_fills_gaps(self):
        """Personal ranking shorter than k gets filled with global."""
        train = pl.DataFrame({
            "user_id": ["u1", "u1"],
            "tv_show_id": [1, 2],
            "is_relevant": [True, True],
            "total_broadcasts_watched": [1, 1],
        })

        eligible = {1, 2, 3, 4, 5}
        rec = PersonalizedFrequencyRecommender(score_by="interaction_count")
        rec.fit(train, eligible)

        # u1 only has 2 personal items, needs 3 more from global
        recs = rec.recommend("u1", 5)
        assert len(recs) == 5
        assert recs[:2] == [1, 2]  # Personal
        # Remaining from global, excluding 1, 2
        assert set(recs[2:]) == {3, 4, 5}

    def test_no_duplicates_in_recommendations(self):
        """Recommendations never contain duplicates."""
        train = pl.DataFrame({
            "user_id": ["u1", "u1", "u2"],
            "tv_show_id": [1, 2, 1],
            "is_relevant": [True, True, True],
            "total_broadcasts_watched": [1, 1, 1],
        })

        eligible = {1, 2, 3}
        rec = PersonalizedFrequencyRecommender(score_by="interaction_count")
        rec.fit(train, eligible)

        recs = rec.recommend("u1", 5)
        assert len(recs) == len(set(recs))  # No duplicates

    def test_score_by_watched_seconds(self):
        """Can score by watched seconds."""
        train = pl.DataFrame({
            "user_id": ["u1", "u1", "u2"],
            "tv_show_id": [1, 2, 1],
            "is_relevant": [True, True, True],
            "total_watched_seconds": [7200, 3600, 1800],
        })

        eligible = {1, 2, 3}
        rec = PersonalizedFrequencyRecommender(score_by="watched_seconds")
        rec.fit(train, eligible)

        # u1: 1(7200), 2(3600) -> [1, 2]
        assert rec.user_rankings["u1"] == [1, 2]


class TestRecencyWeightedRecommender:
    """Tests for recency-weighted baseline."""

    def test_recency_decay(self):
        """Recent interactions weighted higher."""
        import math
        from datetime import datetime, timedelta
        from src.constants import TRAIN_END

        cutoff = TRAIN_END
        recent = cutoff - timedelta(days=1)
        old = cutoff - timedelta(days=30)

        train = pl.DataFrame({
            "user_id": ["u1", "u1"],
            "tv_show_id": [1, 2],
            "is_relevant": [True, True],
            "total_broadcasts_watched": [1, 1],
            "last_interaction_utc": [recent, old],
        })

        eligible = {1, 2}
        rec = RecencyWeightedRecommender(half_life_days=7.0, score_by="interaction_count")
        rec.fit(train, eligible)

        # Recent interaction (1 day ago) should have higher weight than old (30 days ago)
        # Weight = exp(-ln(2) * days / half_life)
        weight_recent = math.exp(-math.log(2) * 1 / 7)
        weight_old = math.exp(-math.log(2) * 30 / 7)
        assert weight_recent > weight_old

        # Programme 1 should rank higher
        assert rec.user_rankings["u1"][0] == 1

    def test_half_life_infinity(self):
        """Very large half-life approximates no decay."""
        import math
        from datetime import datetime, timedelta
        from src.constants import TRAIN_END

        cutoff = TRAIN_END
        recent = cutoff - timedelta(days=1)
        old = cutoff - timedelta(days=30)

        train = pl.DataFrame({
            "user_id": ["u1", "u1"],
            "tv_show_id": [1, 2],
            "is_relevant": [True, True],
            "total_broadcasts_watched": [1, 1],
            "last_interaction_utc": [recent, old],
        })

        eligible = {1, 2}
        rec = RecencyWeightedRecommender(half_life_days=1e9, score_by="interaction_count")
        rec.fit(train, eligible)

        # Both should have nearly equal weights
        # With very large half-life, weights approach 1.0
        # Tie broken by tv_show_id
        assert rec.user_rankings["u1"] == [1, 2]

    def test_fallback_works(self):
        """Cold-start users fall back to global."""
        from src.constants import TRAIN_END
        from datetime import timedelta

        train = pl.DataFrame({
            "user_id": ["u1"],
            "tv_show_id": [1],
            "is_relevant": [True],
            "total_broadcasts_watched": [1],
            "last_interaction_utc": [TRAIN_END - timedelta(days=1)],
        })

        eligible = {1, 2, 3}
        rec = RecencyWeightedRecommender(half_life_days=14.0)
        rec.fit(train, eligible)

        recs = rec.recommend("u2", 3)  # Cold-start user
        assert len(recs) == 3
        assert set(recs) == {1, 2, 3}

    def test_different_half_lives(self):
        """Different half-lives produce different rankings."""
        from datetime import timedelta
        from src.constants import TRAIN_END

        cutoff = TRAIN_END
        recent = cutoff - timedelta(days=1)
        mid = cutoff - timedelta(days=14)
        old = cutoff - timedelta(days=30)

        train = pl.DataFrame({
            "user_id": ["u1", "u1", "u1"],
            "tv_show_id": [1, 2, 3],
            "is_relevant": [True, True, True],
            "total_broadcasts_watched": [1, 1, 1],
            "last_interaction_utc": [recent, mid, old],
        })

        eligible = {1, 2, 3}

        # Short half-life -> recent matters more
        rec_short = RecencyWeightedRecommender(half_life_days=3.0, score_by="interaction_count")
        rec_short.fit(train, eligible)

        # Long half-life -> more equal
        rec_long = RecencyWeightedRecommender(half_life_days=60.0, score_by="interaction_count")
        rec_long.fit(train, eligible)

        # Both should rank 1 first (most recent)
        assert rec_short.user_rankings["u1"][0] == 1
        assert rec_long.user_rankings["u1"][0] == 1


class TestEligibleProgrammes:
    """Tests for eligible programme extraction (schedule-derived, no target leakage)."""

    def test_from_schedule(self):
        """Eligible programmes from schedule."""
        from src.baselines import get_schedule_eligible_programmes
        from datetime import datetime
        import zoneinfo
        
        utc = zoneinfo.ZoneInfo("UTC")
        schedule = pl.DataFrame({
            "airing_id": ["a1", "a2", "a3"],
            "channel_id": [1, 1, 2],
            "programme_start_utc": [
                datetime(2020, 6, 1, 10, 0, 0, tzinfo=utc),
                datetime(2020, 6, 1, 11, 0, 0, tzinfo=utc),
                datetime(2020, 6, 1, 12, 0, 0, tzinfo=utc),
            ],
            "programme_end_utc": [
                datetime(2020, 6, 1, 11, 0, 0, tzinfo=utc),
                datetime(2020, 6, 1, 12, 0, 0, tzinfo=utc),
                datetime(2020, 6, 1, 13, 0, 0, tzinfo=utc),
            ],
            "tv_show_id": [1, 2, 3],
            "programme_duration_seconds": [3600, 3600, 3600],
        })

        window_start = datetime(2020, 6, 1, 10, 30, 0, tzinfo=utc)
        window_end = datetime(2020, 6, 1, 11, 30, 0, tzinfo=utc)

        eligible = get_schedule_eligible_programmes(schedule.lazy(), window_start, window_end)
        # Should include programmes 1 and 2 (air during window), not 3
        assert eligible == {1, 2}

    def test_excludes_unidentified(self):
        """Programme 0 is never eligible."""
        from src.baselines import get_schedule_eligible_programmes
        from datetime import datetime
        import zoneinfo
        
        utc = zoneinfo.ZoneInfo("UTC")
        schedule = pl.DataFrame({
            "airing_id": ["a1", "a2"],
            "channel_id": [1, 1],
            "programme_start_utc": [
                datetime(2020, 6, 1, 10, 0, 0, tzinfo=utc),
                datetime(2020, 6, 1, 11, 0, 0, tzinfo=utc),
            ],
            "programme_end_utc": [
                datetime(2020, 6, 1, 11, 0, 0, tzinfo=utc),
                datetime(2020, 6, 1, 12, 0, 0, tzinfo=utc),
            ],
            "tv_show_id": [0, 1],
            "programme_duration_seconds": [3600, 3600],
        })

        window_start = datetime(2020, 6, 1, 10, 0, 0, tzinfo=utc)
        window_end = datetime(2020, 6, 1, 12, 0, 0, tzinfo=utc)

        eligible = get_schedule_eligible_programmes(schedule.lazy(), window_start, window_end)
        assert 0 not in eligible
        assert eligible == {1}


class TestDeterministicRecommendations:
    """Test that identical inputs produce identical recommendations."""

    def test_global_popularity_deterministic(self):
        """Global popularity is deterministic."""
        train = pl.DataFrame({
            "user_id": ["u1", "u2", "u3"],
            "tv_show_id": [1, 2, 3],
            "is_relevant": [True, True, True],
        })

        eligible = {1, 2, 3}
        rec1 = GlobalPopularityRecommender(score_by="user_count")
        rec1.fit(train, eligible)

        rec2 = GlobalPopularityRecommender(score_by="user_count")
        rec2.fit(train, eligible)

        assert rec1.global_ranking == rec2.global_ranking
        assert rec1.recommend("u1", 5) == rec2.recommend("u1", 5)

    def test_personalized_deterministic(self):
        """Personalized frequency is deterministic."""
        train = pl.DataFrame({
            "user_id": ["u1", "u1", "u2"],
            "tv_show_id": [1, 2, 1],
            "is_relevant": [True, True, True],
            "total_broadcasts_watched": [1, 1, 1],
        })

        eligible = {1, 2, 3}
        rec1 = PersonalizedFrequencyRecommender(score_by="interaction_count")
        rec1.fit(train, eligible)

        rec2 = PersonalizedFrequencyRecommender(score_by="interaction_count")
        rec2.fit(train, eligible)

        assert rec1.user_rankings == rec2.user_rankings
        assert rec1.recommend("u1", 5) == rec2.recommend("u1", 5)

    def test_recency_deterministic(self):
        """Recency-weighted is deterministic."""
        from datetime import timedelta
        from src.constants import TRAIN_END

        train = pl.DataFrame({
            "user_id": ["u1", "u1"],
            "tv_show_id": [1, 2],
            "is_relevant": [True, True],
            "total_broadcasts_watched": [1, 1],
            "last_interaction_utc": [
                TRAIN_END - timedelta(days=1),
                TRAIN_END - timedelta(days=2),
            ],
        })

        eligible = {1, 2}
        rec1 = RecencyWeightedRecommender(half_life_days=7.0)
        rec1.fit(train, eligible)

        rec2 = RecencyWeightedRecommender(half_life_days=7.0)
        rec2.fit(train, eligible)

        assert rec1.user_rankings == rec2.user_rankings
        assert rec1.recommend("u1", 5) == rec2.recommend("u1", 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])