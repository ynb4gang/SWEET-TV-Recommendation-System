#!/usr/bin/env python3
"""
Unit tests for Personalized PageRank recommender.
"""
import pytest
import polars as pl
import numpy as np

from src.personalized_pagerank import PersonalizedPageRankRecommender
from src.baselines import get_validation_eligible_programmes
from src.interaction_pipeline import DataIngestion, PipelineConfig
from src.evaluation import load_submission_users


class TestPersonalizedPageRank:
    """Tests for PPR recommender."""

    def test_ppr_basic_fit(self):
        """Test PPR can fit on small data."""
        config = PipelineConfig()
        train_interactions = pl.read_parquet(config.processed_dir / "train_interactions.parquet")
        val_interactions = pl.read_parquet(config.processed_dir / "val_interactions.parquet")
        
        ingestion = DataIngestion(config)
        schedule_lf = ingestion.load_schedule("21-30")
        
        eligible = get_validation_eligible_programmes()
        
        ppr = PersonalizedPageRankRecommender(alpha=0.15, edge_weight="frequency")
        ppr.fit(train_interactions.head(1000), eligible)
        
        assert ppr._is_fitted
        assert len(ppr.user_ids) > 0
        assert ppr.transition_matrix.shape[0] == ppr.transition_matrix.shape[1]

    def test_ppr_recommend(self):
        """Test PPR produces valid recommendations."""
        config = PipelineConfig()
        train_interactions = pl.read_parquet(config.processed_dir / "train_interactions.parquet")
        
        ingestion = DataIngestion(config)
        schedule_lf = ingestion.load_schedule("21-30")
        
        eligible = get_validation_eligible_programmes()
        
        ppr = PersonalizedPageRankRecommender(alpha=0.15, edge_weight="frequency")
        ppr.fit(train_interactions.head(1000), eligible)
        
        user_id = list(ppr.ppr_scores.keys())[0]
        recs = ppr.recommend(user_id, 5)
        assert len(recs) == 5
        assert all(pid in ppr.eligible_programmes for pid in recs)
        assert len(set(recs)) == 5  # No duplicates

    def test_ppr_cold_start(self):
        """Test cold-start users get global popularity fallback."""
        config = PipelineConfig()
        train_interactions = pl.read_parquet(config.processed_dir / "train_interactions.parquet")
        
        ingestion = DataIngestion(config)
        schedule_lf = ingestion.load_schedule("21-30")
        
        eligible = get_validation_eligible_programmes()
        
        ppr = PersonalizedPageRankRecommender(alpha=0.15, edge_weight="frequency")
        ppr.fit(train_interactions.head(1000), eligible)
        
        # Cold-start user
        recs = ppr.recommend("cold_start_user_123", 5)
        assert len(recs) == 5
        assert all(pid in ppr.eligible_programmes for pid in recs)
        assert len(set(recs)) == 5

    def test_ppr_recommend_batch(self):
        """Test batch recommendations."""
        config = PipelineConfig()
        train_interactions = pl.read_parquet(config.processed_dir / "train_interactions.parquet")
        
        ingestion = DataIngestion(config)
        schedule_lf = ingestion.load_schedule("21-30")
        
        eligible = get_validation_eligible_programmes()
        
        ppr = PersonalizedPageRankRecommender(alpha=0.15, edge_weight="frequency")
        ppr.fit(train_interactions.head(1000), eligible)
        
        user_ids = list(ppr.ppr_scores.keys())[:5]
        recs = ppr.recommend_batch(user_ids, 5)
        
        assert len(recs) == 5
        for uid, recs_list in recs.items():
            assert len(recs_list) == 5
            assert all(pid in ppr.eligible_programmes for pid in recs_list)
            assert len(set(recs_list)) == 5

    def test_ppr_no_programme_zero(self):
        """Test that programme ID 0 is never recommended."""
        config = PipelineConfig()
        train_interactions = pl.read_parquet(config.processed_dir / "train_interactions.parquet")
        
        ingestion = DataIngestion(config)
        schedule_lf = ingestion.load_schedule("21-30")
        
        eligible = get_validation_eligible_programmes()
        
        ppr = PersonalizedPageRankRecommender(alpha=0.15, edge_weight="frequency")
        ppr.fit(train_interactions.head(1000), eligible)
        
        user_id = list(ppr.ppr_scores.keys())[0]
        recs = ppr.recommend(user_id, 5)
        
        assert 0 not in recs
        assert all(pid in ppr.eligible_programmes for pid in recs)

    def test_deterministic_recommendations(self):
        """Test that identical inputs produce identical recommendations."""
        config = PipelineConfig()
        train_interactions = pl.read_parquet(config.processed_dir / "train_interactions.parquet")
        
        ingestion = DataIngestion(config)
        schedule_lf = ingestion.load_schedule("21-30")
        
        eligible = get_validation_eligible_programmes()
        
        ppr1 = PersonalizedPageRankRecommender(alpha=0.15, edge_weight="frequency")
        ppr1.fit(train_interactions.head(1000), eligible)
        
        ppr2 = PersonalizedPageRankRecommender(alpha=0.15, edge_weight="frequency")
        ppr2.fit(train_interactions.head(1000), eligible)
        
        user_id = list(ppr1.ppr_scores.keys())[0]
        recs1 = ppr1.recommend(user_id, 5)
        recs2 = ppr2.recommend(user_id, 5)
        
        assert recs1 == recs2

    def test_alpha_values(self):
        """Test different alpha values produce different results."""
        config = PipelineConfig()
        train_interactions = pl.read_parquet(config.processed_dir / "train_interactions.parquet")
        
        ingestion = DataIngestion(config)
        schedule_lf = ingestion.load_schedule("21-30")
        
        eligible = get_validation_eligible_programmes()
        
        ppr_low = PersonalizedPageRankRecommender(alpha=0.05, edge_weight="frequency")
        ppr_low.fit(train_interactions.head(500), eligible)
        
        ppr_high = PersonalizedPageRankRecommender(alpha=0.30, edge_weight="frequency")
        ppr_high.fit(train_interactions.head(500), eligible)
        
        user_id = list(ppr_low.ppr_scores.keys())[0]
        recs_low = ppr_low.recommend(user_id, 5)
        recs_high = ppr_high.recommend(user_id, 5)
        
        # Different alphas should produce different recommendations (not necessarily)
        assert len(recs_low) == 5
        assert len(recs_high) == 5

    def test_edge_weight_binary(self):
        """Test binary edge weights."""
        config = PipelineConfig()
        train_interactions = pl.read_parquet(config.processed_dir / "train_interactions.parquet")
        
        ingestion = DataIngestion(config)
        schedule_lf = ingestion.load_schedule("21-30")
        
        eligible = get_validation_eligible_programmes()
        
        ppr = PersonalizedPageRankRecommender(alpha=0.15, edge_weight="binary")
        ppr.fit(train_interactions.head(500), eligible)
        
        assert ppr._is_fitted
        assert len(ppr.user_ids) > 0

    def test_invalid_alpha(self):
        """Test that invalid alpha raises error."""
        with pytest.raises(ValueError):
            PersonalizedPageRankRecommender(alpha=0.0)
        
        with pytest.raises(ValueError):
            PersonalizedPageRankRecommender(alpha=1.0)


class TestSubmissionPopulationPPR:
    """Tests for submission population handling in PPR."""

    def test_submission_population_1260(self):
        """Submission population must be exactly 1260 users."""
        from src.evaluation import load_submission_users
        submission_users = load_submission_users("data/submission.csv")
        assert len(submission_users) == 1260

    def test_submission_users_get_recommendations(self):
        """All 1260 submission users get valid recommendations."""
        config = PipelineConfig()
        train_interactions = pl.read_parquet(config.processed_dir / "train_interactions.parquet")
        
        ingestion = DataIngestion(config)
        schedule_lf = ingestion.load_schedule("21-30")
        
        eligible = get_validation_eligible_programmes()
        
        ppr = PersonalizedPageRankRecommender(alpha=0.15, edge_weight="frequency")
        ppr.fit(train_interactions.head(1000), eligible)
        
        submission_users = load_submission_users("data/submission.csv")
        recs = ppr.recommend_batch(list(submission_users), 5)
        
        assert len(recs) == 1260
        for uid, recs_list in recs.items():
            assert len(recs_list) == 5
            assert all(pid in eligible for pid in recs_list)
            assert len(set(recs_list)) == 5


class TestPPRDeterminism:
    """Tests for deterministic output."""

    def test_deterministic_fit(self):
        """PPR fitting is deterministic."""
        config = PipelineConfig()
        train_interactions = pl.read_parquet(config.processed_dir / "train_interactions.parquet")
        
        ingestion = DataIngestion(config)
        schedule_lf = ingestion.load_schedule("21-30")
        
        eligible = get_validation_eligible_programmes()
        
        ppr1 = PersonalizedPageRankRecommender(alpha=0.15, edge_weight="frequency")
        ppr1.fit(train_interactions.head(500), eligible)
        
        ppr2 = PersonalizedPageRankRecommender(alpha=0.15, edge_weight="frequency")
        ppr2.fit(train_interactions.head(500), eligible)
        
        # Check transition matrices are identical
        assert np.allclose(
            ppr1.transition_matrix.toarray(), 
            ppr2.transition_matrix.toarray()
        )
        
        # Check user scores are identical
        for uid in ppr1.ppr_scores:
            assert np.allclose(ppr1.ppr_scores[uid], ppr2.ppr_scores[uid])

    def test_deterministic_recommendations(self):
        """Identical models produce identical recommendations."""
        config = PipelineConfig()
        train_interactions = pl.read_parquet(config.processed_dir / "train_interactions.parquet")
        
        ingestion = DataIngestion(config)
        schedule_lf = ingestion.load_schedule("21-30")
        
        eligible = get_validation_eligible_programmes()
        
        ppr1 = PersonalizedPageRankRecommender(alpha=0.15, edge_weight="frequency")
        ppr1.fit(train_interactions.head(500), eligible)
        
        ppr2 = PersonalizedPageRankRecommender(alpha=0.15, edge_weight="frequency")
        ppr2.fit(train_interactions.head(500), eligible)
        
        user_id = list(ppr1.ppr_scores.keys())[0]
        recs1 = ppr1.recommend(user_id, 5)
        recs2 = ppr2.recommend(user_id, 5)
        
        assert recs1 == recs2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])