"""
Content-Based Filtering recommender for SWEET.TV recommendation system.

Implements metadata-driven programme similarity using sparse TF-IDF vectors.
"""
from __future__ import annotations

import logging
import time
import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .constants import TRAIN_END, VAL_START, VAL_END
from .evaluation import evaluate_model_fixed, get_evaluation_users, load_submission_users
from .baselines import get_validation_eligible_programmes

logger = logging.getLogger(__name__)


class ContentBasedFilteringRecommender:
    """
    Content-Based Filtering recommender using programme metadata.
    
    Builds programme TF-IDF vectors from metadata and user profiles from
    historical viewing, then ranks by cosine similarity.
    """

    def __init__(
        self,
        use_titles: bool = True,
        use_categories: bool = True,
        use_genres: bool = True,
        use_directors: bool = True,
        use_actors: bool = True,
        use_years: bool = False,
        profile_weighting: str = "coverage",  # "positive", "coverage", "frequency"
        title_ngram_range: Tuple[int, int] = (1, 3),
        title_analyzer: str = "char_wb",
        fallback_weight: float = 1.0,
    ):
        """
        Args:
            use_titles: Include programme titles in metadata
            use_categories: Include programme categories
            use_genres: Include genre fields (genre_1, genre_2, genre_3)
            use_directors: Include director names
            use_actors: Include actor names
            use_years: Include production years
            profile_weighting: How to weight programmes in user profile
                "positive" - only is_relevant=True programmes
                "coverage" - weight by coverage ratio
                "frequency" - weight by total_broadcasts_watched
            title_ngram_range: N-gram range for title TF-IDF
            title_analyzer: Analyzer for title TF-IDF ("char_wb", "word")
            fallback_weight: Weight for global popularity fallback
        """
        self.use_titles = use_titles
        self.use_categories = use_categories
        self.use_genres = use_genres
        self.use_directors = use_directors
        self.use_actors = use_actors
        self.use_years = use_years
        self.profile_weighting = profile_weighting
        self.title_ngram_range = title_ngram_range
        self.title_analyzer = title_analyzer
        self.fallback_weight = fallback_weight

        self.programme_vectors: Optional[csr_matrix] = None
        self.programme_ids: List[int] = []
        self.programme_id_to_idx: Dict[int, int] = {}
        self.user_profiles: Dict[str, csr_matrix] = {}
        self.global_popularity_ranking: List[int] = []
        self.eligible_programmes: Set[int] = set()
        self.vectorizers: Dict[str, TfidfVectorizer] = {}
        self._is_fitted = False

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for TF-IDF."""
        if not text or text.strip() == "":
            return ""
        # Normalize whitespace, lowercase
        text = re.sub(r'\s+', ' ', text.strip().lower())
        # Remove special characters but keep alphanumeric and basic punctuation
        text = re.sub(r'[^\w\s\-]', ' ', text)
        # Normalize whitespace again
        text = re.sub(r'\s+', ' ', text.strip())
        return text

    def _extract_programme_text(self, row: Dict) -> Dict[str, str]:
        """Extract and clean text fields from a programme row."""
        texts = {}
        
        def clean_if_valid(val):
            if val is None:
                return ''
            s = str(val).strip()
            if not s or s.lower() in ('null', 'none', 'nan'):
                return ''
            return self._clean_text(s)
        
        if self.use_titles:
            texts['title'] = clean_if_valid(row.get('tv_show_title'))
        
        if self.use_categories:
            texts['category'] = clean_if_valid(row.get('tv_show_category'))
        
        if self.use_genres:
            genre_parts = []
            for g in ['tv_show_genre_1', 'tv_show_genre_2', 'tv_show_genre_3']:
                cleaned = clean_if_valid(row.get(g))
                if cleaned:
                    genre_parts.append(cleaned)
            texts['genres'] = ' '.join(genre_parts)
        
        if self.use_directors:
            texts['director'] = clean_if_valid(row.get('director'))
        
        if self.use_actors:
            texts['actors'] = clean_if_valid(row.get('actors'))
        
        if self.use_years:
            texts['year'] = clean_if_valid(row.get('year_of_production'))
        
        return texts

    def _build_programme_vectors(
        self, 
        schedule_lf: pl.LazyFrame,
        eligible_programmes: Set[int]
    ) -> None:
        """Build TF-IDF vectors for all eligible programmes from schedule metadata."""
        # Get programme metadata for eligible programmes
        schedule_df = schedule_lf.filter(
            pl.col("tv_show_id").is_in(list(eligible_programmes))
        ).collect()
        
        # Sort by tv_show_id for deterministic deduplication
        schedule_df = schedule_df.sort("tv_show_id")
        
        # Deduplicate by tv_show_id (keep first occurrence after sorting)
        metadata_cols = [c for c in ("tv_show_title", "tv_show_category", "tv_show_genre_1",
                         "tv_show_genre_2", "tv_show_genre_3", "director", "actors", "year_of_production")
                         if c in schedule_df.columns]
        schedule_df = schedule_df.sort(["tv_show_id"] + metadata_cols, nulls_last=True)\
            .unique(subset=["tv_show_id"], keep="first", maintain_order=True).sort("tv_show_id")
        
        self.programme_ids = schedule_df["tv_show_id"].to_list()
        self.programme_id_to_idx = {pid: i for i, pid in enumerate(self.programme_ids)}
        
        # Extract text for each field
        n = len(self.programme_ids)
        field_texts = {
            'title': [''] * n,
            'category': [''] * n,
            'genres': [''] * n,
            'director': [''] * n,
            'actors': [''] * n,
            'year': [''] * n,
        }
        
        for i, row in enumerate(schedule_df.iter_rows(named=True)):
            texts = self._extract_programme_text(row)
            for k, v in texts.items():
                if k in field_texts:
                    field_texts[k][i] = v
        
        # Build TF-IDF vectors for each field
        matrices = []
        for field_name, texts in field_texts.items():
            if not any(texts):  # All empty
                continue
            
            if field_name == 'title' and self.use_titles:
                vectorizer = TfidfVectorizer(
                    analyzer=self.title_analyzer,
                    ngram_range=self.title_ngram_range,
                    min_df=1,
                    max_df=0.95,
                    sublinear_tf=True,
                )
            elif field_name in ('category', 'genres') and (self.use_categories or self.use_genres):
                vectorizer = TfidfVectorizer(
                    analyzer='word',
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                    sublinear_tf=True,
                )
            elif field_name in ('director', 'actors') and (self.use_directors or self.use_actors):
                vectorizer = TfidfVectorizer(
                    analyzer='word',
                    ngram_range=(1, 1),
                    min_df=1,
                    max_df=0.95,
                    sublinear_tf=True,
                )
            elif field_name == 'year' and self.use_years:
                vectorizer = TfidfVectorizer(
                    analyzer='word',
                    ngram_range=(1, 1),
                    min_df=1,
                )
            else:
                continue
            
            try:
                tfidf_matrix = vectorizer.fit_transform(texts)
                matrices.append(tfidf_matrix)
                self.vectorizers[field_name] = vectorizer
            except ValueError:
                # Skip if vocabulary is empty
                continue
        
        if matrices:
            # Horizontally stack all field matrices
            self.programme_vectors = hstack(matrices, format='csr')
            # L2 normalize rows (programmes)
            self.programme_vectors = normalize(self.programme_vectors, norm='l2', axis=1)
        else:
            # Fallback: identity matrix
            self.programme_vectors = csr_matrix(np.eye(len(self.programme_ids)))
        
        logger.info(f"Built programme vectors: {self.programme_vectors.shape}")

    def _build_user_profile(self, user_id: str, interactions: pl.DataFrame) -> csr_matrix:
        """Build user content profile from historical interactions."""
        if interactions.height == 0:
            return csr_matrix((1, self.programme_vectors.shape[1]))
        
        # Filter to eligible programmes
        interactions = interactions.filter(
            pl.col("tv_show_id").is_in(self.eligible_programmes)
        )
        
        if interactions.height == 0:
            return csr_matrix((1, self.programme_vectors.shape[1]))
        
        # Get weights based on profile_weighting
        if self.profile_weighting == "positive":
            interactions = interactions.filter(pl.col("is_relevant"))
            weights = np.ones(interactions.height)
        elif self.profile_weighting == "coverage":
            # Use mean_coverage (average coverage across all broadcasts)
            if "mean_coverage_all" in interactions.columns:
                weights = interactions["mean_coverage_all"].to_numpy()
            elif "mean_coverage" in interactions.columns:
                weights = interactions["mean_coverage"].to_numpy()
            elif "mean_coverage_qual" in interactions.columns:
                weights = interactions["mean_coverage_qual"].to_numpy()
            else:
                weights = np.ones(interactions.height)
            weights = np.clip(weights, 0, 1)
        elif self.profile_weighting == "frequency":
            weights = interactions["total_broadcasts_watched"].to_numpy()
            weights = np.log1p(weights)
        else:
            weights = np.ones(interactions.height)
        
        if interactions.height == 0:
            return csr_matrix((1, self.programme_vectors.shape[1]))
        
        # Get programme vectors for user's watched programmes
        programme_indices = []
        programme_weights = []
        
        for i, row in enumerate(interactions.iter_rows(named=True)):
            pid = row["tv_show_id"]
            if pid in self.programme_id_to_idx:
                programme_indices.append(self.programme_id_to_idx[pid])
                programme_weights.append(weights[i] if i < len(weights) else 1.0)
        
        if not programme_indices:
            return csr_matrix((1, self.programme_vectors.shape[1]))
        
        # Weighted average of programme vectors
        user_vector = self.programme_vectors[programme_indices].T @ np.array(programme_weights)
        total_weight = sum(programme_weights)
        if total_weight > 0:
            user_vector = user_vector / total_weight
        
        return csr_matrix(user_vector.reshape(1, -1))

    def fit(
        self,
        train_interactions: pl.DataFrame,
        schedule_lf: pl.LazyFrame,
        eligible_programmes: Set[int],
    ) -> "ContentBasedFilteringRecommender":
        """
        Fit the CBF model on training interactions and schedule metadata.
        
        Args:
            train_interactions: Training user-programme interactions
            schedule_lf: Schedule data (all periods) as LazyFrame
            eligible_programmes: Set of programme IDs eligible for validation
        """
        self.eligible_programmes = eligible_programmes
        
        logger.info("Building programme vectors from schedule metadata...")
        self._build_programme_vectors(schedule_lf, eligible_programmes)
        
        logger.info("Building user profiles from training interactions...")
        train_df = train_interactions.collect() if isinstance(train_interactions, pl.LazyFrame) else train_interactions
        
        # Build global popularity ranking for fallback
        positive_train = train_df.filter(pl.col("is_relevant"))
        global_scores = positive_train.group_by("tv_show_id").agg(
            pl.col("user_id").n_unique().alias("score")
        ).filter(pl.col("tv_show_id").is_in(list(self.eligible_programmes)))
        global_scores = global_scores.sort(["score", "tv_show_id"], descending=[True, False])
        self.global_popularity_ranking = global_scores["tv_show_id"].to_list()
        
        # Ensure ALL eligible programmes are in global ranking
        seen = set(self.global_popularity_ranking)
        for pid in sorted(self.eligible_programmes):
            if pid not in seen:
                self.global_popularity_ranking.append(pid)
        
        # Build user profiles for users with training history
        # Partition by user_id directly
        user_groups = train_df.partition_by("user_id", as_dict=True)
        
        for user_id, group in user_groups.items():
            uid = user_id[0] if isinstance(user_id, tuple) else user_id
            self.user_profiles[uid] = self._build_user_profile(uid, group)
        
        self._is_fitted = True
        logger.info(f"CBF fitted: {len(self.user_profiles)} user profiles, "
                    f"{len(self.programme_ids)} programme vectors")
        return self

    def recommend(self, user_id: str, k: int = 5) -> List[int]:
        """Return top-k programmes for user."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before recommending")
        
        if user_id in self.user_profiles and self.user_profiles[user_id].nnz:
            user_vector = self.user_profiles[user_id]
            
            # Compute cosine similarities
            similarities = self.programme_vectors @ user_vector.T
            similarities = similarities.toarray().flatten()
            
            # Get top-k indices
            top_k_indices = np.lexsort((np.array(self.programme_ids), -similarities))[:k * 3]
            
            # Map back to programme IDs
            recommendations = [self.programme_ids[i] for i in top_k_indices]
            
        else:
            # Cold start: use global popularity
            recommendations = self.global_popularity_ranking
        
        # Ensure exactly k distinct eligible programmes
        final_recs = []
        seen = set()
        for pid in recommendations:
            if pid in self.eligible_programmes and pid not in seen:
                final_recs.append(pid)
                seen.add(pid)
                if len(final_recs) >= k:
                    break
        
        # Fill with global popularity if needed
        if len(final_recs) < k:
            for pid in self.global_popularity_ranking:
                if pid not in seen:
                    final_recs.append(pid)
                    seen.add(pid)
                    if len(final_recs) >= k:
                        break
        
        return final_recs[:k]

    def recommend_batch(self, user_ids: List[str], k: int = 5) -> Dict[str, List[int]]:
        """Return recommendations for multiple users."""
        return {uid: self.recommend(uid, k) for uid in user_ids}


def run_cbf_benchmark(
    train_interactions: pl.DataFrame,
    val_interactions: pl.DataFrame,
    schedule_lf: pl.LazyFrame,
    submission_path: str,
    eligible_programmes: Optional[Set[int]] = None,
) -> Dict:
    """
    Run CBF benchmark with multiple configurations.
    """
    if eligible_programmes is None:
        eligible_programmes = get_validation_eligible_programmes()
    
    submission_users = load_submission_users("data/submission.csv")
    
    logger.info(f"Schedule-derived eligible programmes: {len(eligible_programmes)}")
    
    # Get evaluation populations
    all_users, sub_users, relevant_counts = get_evaluation_users(
        val_interactions, train_interactions, submission_users
    )
    
    logger.info(f"Population A (all known users): {len(all_users)}")
    logger.info(f"Population B (submission users): {len(sub_users)}")
    
    results = {}
    
    # Configuration A: Basic category + genre TF-IDF
    logger.info("Running CBF Configuration A: Basic category + genre TF-IDF")
    cbf_a = ContentBasedFilteringRecommender(
        use_titles=False,
        use_categories=True,
        use_genres=True,
        use_directors=False,
        use_actors=False,
        use_years=False,
        profile_weighting="coverage",
    )
    start = time.perf_counter()
    cbf_a.fit(train_interactions, schedule_lf, eligible_programmes)
    fit_time = time.perf_counter() - start
    
    recs = cbf_a.recommend_batch(all_users, 5)
    stats = evaluate_model_fixed(
        "CBF_Basic_Category_Genre", 
        {u: recs[u] for u in all_users},
        val_interactions, 
        train_interactions,
        submission_users=load_submission_users("data/submission.csv"),
        eligible_programmes=eligible_programmes,
        k=5
    )
    stats["runtime_seconds"] = time.perf_counter() - start
    stats["fallback_percentage"] = 0.0  # TODO: compute actual
    results["CBF_Basic_Category_Genre"] = stats
    
    # Configuration B: Rich metadata TF-IDF (including titles)
    logger.info("Running CBF Configuration B: Rich metadata TF-IDF")
    cbf_b = ContentBasedFilteringRecommender(
        use_titles=True,
        use_categories=True,
        use_genres=True,
        use_directors=True,
        use_actors=True,
        use_years=False,
        profile_weighting="coverage",
        title_ngram_range=(1, 3),
        title_analyzer="char_wb",
    )
    start = time.perf_counter()
    cbf_b.fit(train_interactions, schedule_lf, eligible_programmes)
    fit_time = time.perf_counter() - start
    
    recs = cbf_b.recommend_batch(all_users, 5)
    stats = evaluate_model_fixed(
        "CBF_Rich_Metadata", 
        {u: recs[u] for u in all_users},
        val_interactions, 
        train_interactions,
        submission_users=load_submission_users("data/submission.csv"),
        eligible_programmes=eligible_programmes,
        k=5
    )
    stats["runtime_seconds"] = time.perf_counter() - start
    results["CBF_Rich_Metadata"] = stats
    
    # Configuration C: Rich metadata with frequency-weighted profiles
    logger.info("Running CBF Configuration C: Rich metadata + frequency-weighted profiles")
    cbf_c = ContentBasedFilteringRecommender(
        use_titles=True,
        use_categories=True,
        use_genres=True,
        use_directors=True,
        use_actors=True,
        use_years=False,
        profile_weighting="frequency",
        title_ngram_range=(1, 3),
        title_analyzer="char_wb",
    )
    start = time.perf_counter()
    cbf_c.fit(train_interactions, schedule_lf, eligible_programmes)
    fit_time = time.perf_counter() - start
    
    recs = cbf_c.recommend_batch(all_users, 5)
    stats = evaluate_model_fixed(
        "CBF_Rich_Frequency_Weighted", 
        {u: recs[u] for u in all_users},
        val_interactions, 
        train_interactions,
        submission_users=load_submission_users("data/submission.csv"),
        eligible_programmes=eligible_programmes,
        k=5
    )
    stats["runtime_seconds"] = time.perf_counter() - start
    results["CBF_Rich_Frequency_Weighted"] = stats
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from src.constants import PipelineConfig
    from src.interaction_pipeline import PipelineConfig as IPConfig
    from src.interaction_pipeline import run_pipeline
    
    # Load interactions
    config = IPConfig()
    train_path = config.processed_dir / "train_interactions.parquet"
    val_path = config.processed_dir / "val_interactions.parquet"
    
    train_interactions = pl.read_parquet(train_path)
    val_interactions = pl.read_parquet(val_path)
    
    # Load schedule
    from src.interaction_pipeline import DataIngestion
    ingestion = DataIngestion(config)
    schedule_lf = ingestion.load_all_schedules()
    
    results = run_cbf_benchmark(train_interactions, val_interactions, schedule_lf, "data/submission.csv")
    
    import json
    with open("evidence/phase04_cbf.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(json.dumps(results, indent=2))
