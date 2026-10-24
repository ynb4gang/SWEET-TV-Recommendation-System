#!/usr/bin/env python3
"""
Unit tests for Content-Based Filtering recommender.
"""
import pytest
import polars as pl
import numpy as np
from datetime import datetime
import zoneinfo
from sklearn.feature_extraction.text import TfidfVectorizer

from src.content_based import (
    ContentBasedFilteringRecommender,
    run_cbf_benchmark,
)
from src.baselines import get_validation_eligible_programmes
from src.interaction_pipeline import DataIngestion, PipelineConfig
from src.evaluation import load_submission_users


class TestContentBasedFiltering:
    """Tests for CBF recommender."""

    def test_text_cleaning(self):
        """Test text cleaning function."""
        cbf = ContentBasedFilteringRecommender()
        assert cbf._clean_text("  Hello World  ") == "hello world"
        assert cbf._clean_text("Title with SPECIAL chars!@#") == "title with special chars"
        assert cbf._clean_text("") == ""
        assert cbf._clean_text("   ") == ""

    def test_programme_text_extraction(self):
        """Test extracting text from programme metadata."""
        cbf = ContentBasedFilteringRecommender(
            use_titles=True,
            use_categories=True,
            use_genres=True,
            use_directors=True,
            use_actors=True,
            use_years=True,
        )
        
        row = {
            'tv_show_title': '  The Matrix  ',
            'tv_show_category': 'Movie',
            'tv_show_genre_1': 'Action',
            'tv_show_genre_2': 'Sci-Fi',
            'tv_show_genre_3': 'null',
            'director': 'Lana Wachowski',
            'actors': 'Keanu Reeves, Laurence Fishburne',
            'year_of_production': '1999',
        }
        
        texts = cbf._extract_programme_text(row)
        assert texts['title'] == 'the matrix'
        assert texts['category'] == 'movie'
        assert texts['genres'] == 'action sci-fi'
        assert texts['director'] == 'lana wachowski'
        # Comma is removed by text cleaning, so expect space-separated
        assert texts['actors'] == 'keanu reeves laurence fishburne'
        assert texts['year'] == '1999'

    def test_null_metadata_handling(self):
        """Test handling of null/missing metadata."""
        cbf = ContentBasedFilteringRecommender(
            use_titles=True,
            use_categories=True,
            use_genres=True,
            use_directors=True,
            use_actors=True,
            use_years=True,
        )
        
        row = {
            'tv_show_title': None,
            'tv_show_category': 'null',
            'tv_show_genre_1': '',
            'tv_show_genre_2': 'null',
            'tv_show_genre_3': 'nan',
            'director': 'none',
            'actors': 'None',
            'year_of_production': 'NaN',
        }
        
        texts = cbf._extract_programme_text(row)
        assert texts['title'] == ''
        assert texts['category'] == ''
        assert texts['genres'] == ''
        assert texts['director'] == ''
        assert texts['actors'] == ''
        assert texts['year'] == ''

    def test_cosine_similarity_ranking(self):
        """Test that cosine similarity produces correct rankings."""
        from scipy.sparse import csr_matrix
        from sklearn.preprocessing import normalize
        
        # Create simple test vectors
        # Item 0: [1, 0, 0]
        # Item 1: [0, 1, 0]
        # Item 2: [0.707, 0.707, 0] (45 degrees from both)
        vectors = csr_matrix([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.707, 0.707, 0.0],
        ])
        vectors = normalize(vectors, norm='l2', axis=1)
        
        # Query vector: [1, 0, 0] (same as item 0)
        query = csr_matrix([[1.0, 0.0, 0.0]])
        query = normalize(query, norm='l2', axis=1)
        
        # Similarities: item 0 = 1.0, item 2 = 0.707, item 1 = 0
        similarities = vectors @ query.T
        sims = similarities.toarray().flatten()
        
        assert np.argmax(sims) == 0  # Item 0 is most similar
        assert similarities[0, 0] > similarities[2, 0] > similarities[1, 0]


class TestCBFIntegration:
    """Integration tests with real data (smoke tests)."""

    def test_cbf_basic_fit_and_recommend(self):
        """Test CBF can fit and produce recommendations."""
        config = PipelineConfig()
        train_interactions = pl.read_parquet(config.processed_dir / "train_interactions.parquet")
        val_interactions = pl.read_parquet(config.processed_dir / "val_interactions.parquet")
        
        ingestion = DataIngestion(config)
        schedule_lf = ingestion.load_schedule("21-30")
        
        eligible = get_validation_eligible_programmes()
        
        cbf = ContentBasedFilteringRecommender(
            use_titles=False,
            use_categories=True,
            use_genres=True,
            use_directors=False,
            use_actors=False,
            profile_weighting="coverage",
        )
        
        cbf.fit(train_interactions.head(10000), schedule_lf, eligible)
        
        assert cbf._is_fitted
        assert len(cbf.user_profiles) > 0
        assert cbf.programme_vectors.shape[0] > 0
        
        # Test recommendations
        user_id = list(cbf.user_profiles.keys())[0]
        recs = cbf.recommend(user_id, 5)
        assert len(recs) == 5
        assert all(pid in cbf.eligible_programmes for pid in recs)
        assert len(set(recs)) == 5  # No duplicates

    def test_cold_start_recommendations(self):
        """Test cold-start users get global popularity fallback."""
        config = PipelineConfig()
        train_interactions = pl.read_parquet(config.processed_dir / "train_interactions.parquet")
        
        ingestion = DataIngestion(config)
        schedule_lf = ingestion.load_schedule("21-30")
        
        eligible = get_validation_eligible_programmes()
        
        cbf = ContentBasedFilteringRecommender(
            use_titles=False,
            use_categories=True,
            use_genres=True,
        )
        
        cbf.fit(train_interactions.head(1000), schedule_lf, eligible)
        
        # Cold-start user (not in training)
        recs = cbf.recommend("cold_start_user_123", 5)
        assert len(recs) == 5
        assert all(pid in cbf.eligible_programmes for pid in recs)
        assert len(set(recs)) == 5

    def test_recommend_batch(self):
        """Test batch recommendations."""
        config = PipelineConfig()
        train_interactions = pl.read_parquet(config.processed_dir / "train_interactions.parquet")
        
        ingestion = DataIngestion(config)
        schedule_lf = ingestion.load_schedule("21-30")
        
        eligible = get_validation_eligible_programmes()
        
        cbf = ContentBasedFilteringRecommender(
            use_titles=False,
            use_categories=True,
            use_genres=True,
        )
        
        cbf.fit(train_interactions.head(1000), schedule_lf, eligible)
        
        user_ids = list(cbf.user_profiles.keys())[:5]
        recs = cbf.recommend_batch(user_ids, 5)
        
        assert len(recs) == 5
        for uid, recs_list in recs.items():
            assert len(recs_list) == 5
            assert all(pid in eligible for pid in recs_list)
            assert len(set(recs_list)) == 5

    def test_deterministic_recommendations(self):
        """Test that identical inputs produce identical recommendations."""
        config = PipelineConfig()
        train_interactions = pl.read_parquet(config.processed_dir / "train_interactions.parquet")
        
        ingestion = DataIngestion(config)
        schedule_lf = ingestion.load_schedule("21-30")
        
        eligible = get_validation_eligible_programmes()
        
        cbf1 = ContentBasedFilteringRecommender(
            use_titles=False,
            use_categories=True,
            use_genres=True,
        )
        cbf1.fit(train_interactions.head(1000), schedule_lf, eligible)
        
        cbf2 = ContentBasedFilteringRecommender(
            use_titles=False,
            use_categories=True,
            use_genres=True,
        )
        cbf2.fit(train_interactions.head(1000), schedule_lf, eligible)
        
        user_id = list(cbf1.user_profiles.keys())[0]
        recs1 = cbf1.recommend(user_id, 5)
        recs2 = cbf2.recommend(user_id, 5)
        
        assert recs1 == recs2

    def test_no_programme_zero(self):
        """Test that programme ID 0 is never recommended."""
        config = PipelineConfig()
        train_interactions = pl.read_parquet(config.processed_dir / "train_interactions.parquet")
        
        ingestion = DataIngestion(config)
        schedule_lf = ingestion.load_schedule("21-30")
        
        eligible = get_validation_eligible_programmes()
        
        cbf = ContentBasedFilteringRecommender(
            use_titles=False,
            use_categories=True,
            use_genres=True,
        )
        
        cbf.fit(train_interactions.head(1000), schedule_lf, eligible)
        
        user_id = list(cbf.user_profiles.keys())[0]
        recs = cbf.recommend(user_id, 5)
        
        assert 0 not in recs
        assert all(pid in cbf.eligible_programmes for pid in recs)


class TestCBFConfigurations:
    """Test different CBF configurations."""

    def test_config_a_basic(self):
        """Test Configuration A: Basic category + genre TF-IDF."""
        cbf = ContentBasedFilteringRecommender(
            use_titles=False,
            use_categories=True,
            use_genres=True,
            use_directors=False,
            use_actors=False,
            use_years=False,
            profile_weighting="coverage",
        )
        assert cbf.use_titles == False
        assert cbf.use_categories == True
        assert cbf.use_genres == True
        assert cbf.profile_weighting == "coverage"

    def test_config_b_rich(self):
        """Test Configuration B: Rich metadata TF-IDF."""
        cbf = ContentBasedFilteringRecommender(
            use_titles=True,
            use_categories=True,
            use_genres=True,
            use_directors=True,
            use_actors=True,
            use_years=False,
            profile_weighting="coverage",
        )
        assert cbf.use_titles == True
        assert cbf.use_directors == True
        assert cbf.use_actors == True

    def test_config_c_frequency_weighted(self):
        """Test Configuration C: Frequency-weighted profiles."""
        cbf = ContentBasedFilteringRecommender(
            use_titles=True,
            use_categories=True,
            use_genres=True,
            use_directors=True,
            use_actors=True,
            use_years=False,
            profile_weighting="frequency",
        )
        assert cbf.profile_weighting == "frequency"


class TestSubmissionPopulation:
    """Tests for submission population handling."""

    def test_submission_population_1260(self):
        """Submission population must be exactly 1260 users."""
        from src.evaluation import load_submission_users
        submission_users = load_submission_users("data/submission.csv")
        assert len(submission_users) == 1260

    def test_submission_users_not_intersected(self):
        """Submission users must NOT be intersected with training users."""
        from src.evaluation import load_submission_users
        from src.content_based import get_evaluation_users
        
        config = PipelineConfig()
        train_interactions = pl.read_parquet(config.processed_dir / "train_interactions.parquet")
        val_interactions = pl.read_parquet(config.processed_dir / "val_interactions.parquet")
        
        submission_users = load_submission_users("data/submission.csv")
        all_users, sub_users, counts = get_evaluation_users(
            val_interactions, train_interactions, submission_users
        )
        
        assert len(sub_users) == 1260
        assert set(sub_users) == submission_users
        # Not all submission users may be in training
        train_users = set(train_interactions.select("user_id").unique().to_series().to_list())
        assert not set(sub_users) <= train_users  # Not all submission users in training

    def test_cold_start_submission_users_get_recommendations(self):
        """Submission users with no training history get valid recommendations."""
        config = PipelineConfig()
        train_interactions = pl.read_parquet(config.processed_dir / "train_interactions.parquet")
        
        ingestion = DataIngestion(config)
        schedule_lf = ingestion.load_schedule("21-30")
        
        eligible = get_validation_eligible_programmes()
        
        cbf = ContentBasedFilteringRecommender(
            use_titles=False,
            use_categories=True,
            use_genres=True,
        )
        
        cbf.fit(train_interactions.head(1000), schedule_lf, eligible)
        
        submission_users = load_submission_users("data/submission.csv")
        # Find a submission user NOT in training
        train_users = set(train_interactions.select("user_id").unique().to_series().to_list())
        cold_start_users = [u for u in submission_users if u not in train_users]
        
        if cold_start_users:
            user_id = cold_start_users[0]
            recs = cbf.recommend(user_id, 5)
            assert len(recs) == 5
            assert all(pid in eligible for pid in recs)


class TestRecommendationQuality:
    """Tests for recommendation quality guarantees."""

    def test_exactly_five_recommendations(self):
        """Every user gets exactly 5 recommendations."""
        config = PipelineConfig()
        train_interactions = pl.read_parquet(config.processed_dir / "train_interactions.parquet")
        
        ingestion = DataIngestion(config)
        schedule_lf = ingestion.load_schedule("21-30")
        
        eligible = get_validation_eligible_programmes()
        
        cbf = ContentBasedFilteringRecommender(
            use_titles=False,
            use_categories=True,
            use_genres=True,
        )
        
        cbf.fit(train_interactions.head(1000), schedule_lf, eligible)
        
        user_id = list(cbf.user_profiles.keys())[0]
        recs = cbf.recommend(user_id, 5)
        assert len(recs) == 5

    def test_no_duplicate_recommendations(self):
        """Recommendations never contain duplicates."""
        config = PipelineConfig()
        train_interactions = pl.read_parquet(config.processed_dir / "train_interactions.parquet")
        
        ingestion = DataIngestion(config)
        schedule_lf = ingestion.load_schedule("21-30")
        
        eligible = get_validation_eligible_programmes()
        
        cbf = ContentBasedFilteringRecommender(
            use_titles=False,
            use_categories=True,
            use_genres=True,
        )
        
        cbf.fit(train_interactions.head(1000), schedule_lf, eligible)
        
        user_id = list(cbf.user_profiles.keys())[0]
        recs = cbf.recommend(user_id, 5)
        assert len(recs) == len(set(recs))

    def test_only_eligible_programmes_recommended(self):
        """Only eligible programmes are recommended."""
        config = PipelineConfig()
        train_interactions = pl.read_parquet(config.processed_dir / "train_interactions.parquet")
        
        ingestion = DataIngestion(config)
        schedule_lf = ingestion.load_schedule("21-30")
        
        eligible = get_validation_eligible_programmes()
        
        cbf = ContentBasedFilteringRecommender(
            use_titles=False,
            use_categories=True,
            use_genres=True,
        )
        
        cbf.fit(train_interactions.head(1000), schedule_lf, eligible)
        
        user_id = list(cbf.user_profiles.keys())[0]
        recs = cbf.recommend(user_id, 5)
        
        for pid in recs:
            assert pid in eligible


class TestCBFDeterminism:
    """Tests for deterministic output."""

    def test_deterministic_fit(self):
        """Identical training produces identical models."""
        config = PipelineConfig()
        train_interactions = pl.read_parquet(config.processed_dir / "train_interactions.parquet")
        
        ingestion = DataIngestion(config)
        schedule_lf = ingestion.load_schedule("21-30")
        
        eligible = get_validation_eligible_programmes()
        
        cbf1 = ContentBasedFilteringRecommender(
            use_titles=False,
            use_categories=True,
            use_genres=True,
        )
        cbf1.fit(train_interactions.head(1000), schedule_lf, eligible)
        
        cbf2 = ContentBasedFilteringRecommender(
            use_titles=False,
            use_categories=True,
            use_genres=True,
        )
        cbf2.fit(train_interactions.head(1000), schedule_lf, eligible)
        
        # Check programme vectors are identical
        assert np.allclose(
            cbf1.programme_vectors.toarray(), 
            cbf2.programme_vectors.toarray()
        )

    def test_deterministic_recommendations(self):
        """Identical models produce identical recommendations."""
        config = PipelineConfig()
        train_interactions = pl.read_parquet(config.processed_dir / "train_interactions.parquet")
        
        ingestion = DataIngestion(config)
        schedule_lf = ingestion.load_schedule("21-30")
        
        eligible = get_validation_eligible_programmes()
        
        cbf1 = ContentBasedFilteringRecommender(
            use_titles=False,
            use_categories=True,
            use_genres=True,
        )
        cbf1.fit(train_interactions.head(1000), schedule_lf, eligible)
        
        cbf2 = ContentBasedFilteringRecommender(
            use_titles=False,
            use_categories=True,
            use_genres=True,
        )
        cbf2.fit(train_interactions.head(1000), schedule_lf, eligible)
        
        user_id = list(cbf1.user_profiles.keys())[0]
        recs1 = cbf1.recommend(user_id, 5)
        recs2 = cbf2.recommend(user_id, 5)
        
        assert recs1 == recs2


class TestMetadataVectorization:
    """Tests for metadata vectorization."""

    def test_tfidf_vectorization(self):
        """Test TF-IDF vectorization works correctly."""
        texts = [
            "action movie",
            "comedy movie",
            "action thriller",
            "drama series",
        ]
        
        vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            min_df=1,
        )
        tfidf = vectorizer.fit_transform(texts)
        
        assert tfidf.shape == (4, len(vectorizer.vocabulary_))
        assert tfidf.nnz > 0
        
        # Check L2 normalization
        from sklearn.preprocessing import normalize
        normalized = normalize(tfidf, norm='l2', axis=1)
        norms = np.sqrt(normalized.multiply(normalized).sum(axis=1))
        assert np.allclose(norms, 1.0)

    def test_char_ngram_vectorization(self):
        """Test character n-gram vectorization for titles."""
        titles = [
            "the matrix",
            "matrix reloaded",
            "the godfather",
        ]
        
        vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(1, 3),
            min_df=1,
        )
        tfidf = vectorizer.fit_transform(titles)
        
        assert tfidf.shape[0] == 3
        assert tfidf.nnz > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])