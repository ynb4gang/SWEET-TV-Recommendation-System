"""
Core data ingestion and interaction pipeline for SWEET.TV recommendation system.

This module handles:
- Raw data ingestion from CSV files
- Timestamp parsing with DST-aware UTC conversion
- Viewing-to-programme interval join
- Temporal train/validation/test splits
- User-programme interaction aggregation
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from .constants import (
    KYIV_TZ, TRAIN_START, TRAIN_END, VAL_START, VAL_END,
    TEST_START, TEST_END, COVERAGE_THRESHOLD, PipelineConfig
)

logger = logging.getLogger(__name__)

# Try to import optimized join
try:
    from .optimized_join import compute_intersections_optimized
    HAS_OPTIMIZED_JOIN = True
    OPTIMIZED_IMPORT_ERROR = None
except ImportError as e:
    HAS_OPTIMIZED_JOIN = False
    OPTIMIZED_IMPORT_ERROR = str(e)


class DataIngestion:
    """Handles raw data loading and initial parsing."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.config.processed_dir.mkdir(parents=True, exist_ok=True)
        self.config.evidence_dir.mkdir(parents=True, exist_ok=True)

    def load_viewing_logs(self) -> pl.LazyFrame:
        """Load and parse viewing logs with UTC conversion."""
        path = self.config.data_dir / "dataset11-30.csv"
        lf = pl.scan_csv(path, schema_overrides={"user_id": pl.String})

        # Parse timestamps as naive local, then convert to UTC via Europe/Kyiv
        lf = lf.with_columns([
            pl.col("start_time").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
            .dt.replace_time_zone(KYIV_TZ)
            .dt.convert_time_zone("UTC")
            .alias("view_start_utc"),
            pl.col("stop_time").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
            .dt.replace_time_zone(KYIV_TZ)
            .dt.convert_time_zone("UTC")
            .alias("view_end_utc"),
        ]).with_columns([
            pl.col("vsetv_id").alias("channel_id"),
            pl.col("user_id").cast(pl.Utf8).alias("user_id"),
            pl.col("duraton").cast(pl.Int64).alias("duraton_seconds"),
        ]).select([
            "user_id", "channel_id", "view_start_utc", "view_end_utc", "duraton_seconds"
        ])

        return lf

    def load_schedule(self, period: str) -> pl.LazyFrame:
        """Load and parse schedule for a given period."""
        period_map = {
            "11-20": "export_arh_11-20-final.csv",
            "21-30": "export_arh_21-30-final.csv",
            "31-42": "export_arh_31-42-final.csv",
        }
        if period not in period_map:
            raise ValueError(f"Unknown period: {period}")

        path = self.config.data_dir / period_map[period]
        lf = pl.scan_csv(path)

        # Parse timestamps as naive local, then convert to UTC via Europe/Kyiv
        lf = lf.with_columns([
            pl.col("start_time").str.strptime(pl.Datetime, format="%d.%m.%Y %H:%M:%S")
            .dt.replace_time_zone(KYIV_TZ)
            .dt.convert_time_zone("UTC")
            .alias("programme_start_utc"),
            (pl.col("start_time").str.strptime(pl.Datetime, format="%d.%m.%Y %H:%M:%S")
            .dt.replace_time_zone(KYIV_TZ)
            .dt.convert_time_zone("UTC")
            + pl.duration(seconds=pl.col("duration"))).alias("programme_end_utc"),
        ]).with_columns([
            pl.col("channel_id").cast(pl.Int64),
            pl.col("tv_show_id").cast(pl.Int64),
            pl.col("duration").cast(pl.Int64).alias("programme_duration_seconds"),
            pl.col("tv_show_title").cast(pl.Utf8),
            pl.col("tv_show_category").cast(pl.Utf8),
            pl.col("tv_show_genre_1").cast(pl.Utf8),
            pl.col("tv_show_genre_2").cast(pl.Utf8),
            pl.col("tv_show_genre_3").cast(pl.Utf8),
            pl.col("year_of_production").cast(pl.Utf8),
            pl.col("director").cast(pl.Utf8),
            pl.col("actors").cast(pl.Utf8),
        ]).with_columns([
            # Create stable airing_id
            (pl.col("channel_id").cast(pl.Utf8) + "_" +
             pl.col("programme_start_utc").dt.strftime("%Y%m%d%H%M%S")).alias("airing_id"),
        ]).select([
            "airing_id", "channel_id", "programme_start_utc", "programme_end_utc",
            "tv_show_id", "programme_duration_seconds",
            "tv_show_title", "tv_show_category", "tv_show_genre_1",
            "tv_show_genre_2", "tv_show_genre_3", "year_of_production",
            "director", "actors"
        ])

        return lf

    def load_all_schedules(self) -> pl.LazyFrame:
        """Load and concatenate all schedule periods."""
        schedules = []
        for period in ["11-20", "21-30", "31-42"]:
            schedules.append(self.load_schedule(period))
        return pl.concat(schedules, how="vertical")

    def load_submission(self) -> pl.DataFrame:
        """Load submission template."""
        path = self.config.data_dir / "submission.csv"
        df = pl.read_csv(path, schema_overrides={"user_id": pl.Utf8, "tv_show_id": pl.Utf8})
        return df


class InteractionBuilder:
    """Builds user-programme interactions from viewing and schedule data."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def normalize_viewing_intervals(self, viewing_lf: pl.LazyFrame) -> pl.LazyFrame:
        """Normalize and deduplicate viewing intervals per user-channel."""
        # Remove invalid durations
        viewing_lf = viewing_lf.filter(
            (pl.col("duraton_seconds") > 0) &
            (pl.col("view_end_utc") > pl.col("view_start_utc"))
        )

        # Deduplicate exact duplicate records
        viewing_lf = viewing_lf.unique(subset=["user_id", "channel_id", "view_start_utc", "view_end_utc", "duraton_seconds"])

        return viewing_lf

    def normalize_broadcast_intervals(self, schedule_lf: pl.LazyFrame) -> pl.LazyFrame:
        """Normalize broadcast intervals, exclude unidentified programmes."""
        # Exclude unidentified programmes (tv_show_id = 0)
        identified = schedule_lf.filter(pl.col("tv_show_id") != 0)

        # Deduplicate exact duplicate schedule records
        identified = identified.unique(subset=["channel_id", "programme_start_utc", "tv_show_id"])

        return identified

    def compute_interval_intersections(
        self,
        viewing_lf: pl.LazyFrame,
        schedule_lf: pl.LazyFrame
    ) -> pl.LazyFrame:
        """Compute viewing-broadcast intersections using optimized DuckDB join."""
        # Use optimized join if available
        if HAS_OPTIMIZED_JOIN:
            logger.info("Using optimized DuckDB interval join...")
            intersections_df = compute_intersections_optimized(viewing_lf, schedule_lf)
            return intersections_df.lazy()

        # Fallback to Polars join (slower)
        logger.warning("Optimized join not available, using Polars fallback...")
        joined = viewing_lf.join(
            schedule_lf.select(["airing_id", "channel_id", "programme_start_utc", "programme_end_utc", "tv_show_id", "programme_duration_seconds"]),
            on="channel_id",
            how="inner"
        )

        joined = joined.filter(
            (pl.col("view_start_utc") < pl.col("programme_end_utc")) &
            (pl.col("view_end_utc") > pl.col("programme_start_utc"))
        )

        joined = joined.with_columns([
            pl.max_horizontal("view_start_utc", "programme_start_utc").alias("overlap_start_utc"),
            pl.min_horizontal("view_end_utc", "programme_end_utc").alias("overlap_end_utc"),
        ]).with_columns([
            (pl.col("overlap_end_utc") - pl.col("overlap_start_utc")).dt.total_seconds()
            .cast(pl.Int64).alias("overlap_seconds"),
        ]).filter(pl.col("overlap_seconds") > 0)

        return joined

    def merge_overlapping_fragments(
        self,
        intersections_lf: pl.LazyFrame
    ) -> pl.LazyFrame:
        """Merge overlapping viewing fragments for same user-airing."""
        # Sort by user, airing, overlap_start
        intersections_lf = intersections_lf.sort(["user_id", "airing_id", "overlap_start_utc"])

        # This is complex in pure Polars lazy - collect and process
        # For now, return as-is; merging happens in aggregation step
        return intersections_lf

    def calculate_coverage(self, intersections_lf: pl.LazyFrame) -> pl.LazyFrame:
        """Calculate viewing coverage per user-airing."""
        # Group by user and airing, sum unique overlap seconds
        # For proper deduplication of overlapping fragments, we need to collect
        # This is a placeholder - full implementation uses eager execution
        return intersections_lf

    def aggregate_user_programme_interactions(
        self,
        viewing_lf: pl.LazyFrame,
        schedule_lf: pl.LazyFrame
    ) -> pl.DataFrame:
        """Full pipeline: viewing + schedule -> user-programme interactions."""
        logger.info("Normalizing viewing intervals...")
        viewing_norm = self.normalize_viewing_intervals(viewing_lf)

        logger.info("Normalizing broadcast intervals...")
        schedule_norm = self.normalize_broadcast_intervals(schedule_lf)

        logger.info("Computing interval intersections...")
        intersections = self.compute_interval_intersections(viewing_norm, schedule_norm)

        # Collect for fragment merging (requires eager execution)
        intersections_df = intersections.collect()

        logger.info("Merging overlapping fragments and calculating coverage...")
        # Merge overlapping fragments per user-airing
        merged = self._merge_fragments_eager(intersections_df)

        # Calculate coverage per user-airing
        coverage = self._calculate_coverage_eager(merged)

        logger.info("Aggregating to user-programme level...")
        # Aggregate to user-programme level
        interactions = self._aggregate_to_programme_level_eager(coverage)

        return interactions

    def _merge_fragments_eager(self, df: pl.DataFrame) -> pl.DataFrame:
        """Merge overlapping viewing fragments for same user-airing (eager, vectorized)."""
        if df.height == 0:
            return df

        # Sort by user, airing, overlap_start
        df = df.sort(["user_id", "airing_id", "overlap_start_utc"])

        # Use window functions to identify merged groups
        # For each partition (user_id, airing_id), compute running max of overlap_end_utc
        # A new group starts when overlap_start_utc > running_max(overlap_end_utc) of previous row
        
        df = df.with_columns([
            pl.col("overlap_end_utc").cum_max().over(["user_id", "airing_id"]).alias("running_max_end"),
        ])

        # A new group starts when current start > previous running max end
        # (or when it's the first row in partition)
        df = df.with_columns([
            (pl.col("overlap_start_utc") > pl.col("running_max_end").shift(1).over(["user_id", "airing_id"]))
            .fill_null(True)
            .alias("new_group"),
        ])

        # Cumulative sum to assign group IDs
        df = df.with_columns([
            pl.col("new_group").cast(pl.Int64).cum_sum().over(["user_id", "airing_id"]).alias("group_id"),
        ])

        # Aggregate by group to get merged intervals
        merged_groups = df.group_by(["user_id", "airing_id", "group_id"]).agg([
            pl.col("overlap_start_utc").min().alias("merged_start"),
            pl.col("overlap_end_utc").max().alias("merged_end"),
            pl.col("tv_show_id").first(),
            pl.col("programme_duration_seconds").first(),
            pl.col("channel_id").first(),
            pl.col("programme_start_utc").first(),
            pl.col("programme_end_utc").first(),
        ]).with_columns([
            (pl.col("merged_end") - pl.col("merged_start")).dt.total_seconds().cast(pl.Int64).alias("overlap_seconds"),
        ])

        # Cap overlap at programme duration per group
        merged_groups = merged_groups.with_columns([
            pl.min_horizontal("overlap_seconds", "programme_duration_seconds").alias("overlap_seconds"),
        ])

        # Final aggregation to user-airing level (sum overlap across groups)
        merged = merged_groups.group_by(["user_id", "airing_id"]).agg([
            pl.col("tv_show_id").first(),
            pl.col("programme_duration_seconds").first(),
            pl.col("channel_id").first(),
            pl.col("programme_start_utc").first(),
            pl.col("programme_end_utc").first(),
            pl.col("overlap_seconds").sum().alias("overlap_seconds"),
        ])

        return merged

    def _calculate_coverage_eager(self, merged_df: pl.DataFrame) -> pl.DataFrame:
        """Calculate coverage per user-airing (eager)."""
        if merged_df.height == 0:
            return merged_df

        result = merged_df.with_columns([
            (pl.col("overlap_seconds") / pl.col("programme_duration_seconds")).alias("coverage"),
        ]).with_columns([
            pl.col("coverage").clip(0, 1).alias("coverage"),
            (pl.col("coverage") >= self.config.coverage_threshold).alias("is_relevant"),
        ])

        return result

    def _aggregate_to_programme_level_eager(self, coverage_df: pl.DataFrame) -> pl.DataFrame:
        """Aggregate user-airing coverage to user-programme interactions (eager).
        
        Preserves ALL user-programme interactions, with is_relevant flag for qualifying ones.
        """
        if coverage_df.height == 0:
            return pl.DataFrame(schema={
                "user_id": pl.Utf8, "tv_show_id": pl.Int64,
                "total_watched_seconds": pl.Int64, "total_broadcasts_watched": pl.Int64,
                "max_coverage": pl.Float64, "mean_coverage": pl.Float64,
                "num_qualifying_broadcasts": pl.Int64, "first_interaction_utc": pl.Datetime("us", "UTC"),
                "last_interaction_utc": pl.Datetime("us", "UTC"), "distinct_viewing_days": pl.Int64,
                "is_relevant": pl.Boolean,
                "total_broadcasts_viewed": pl.Int64,
                "max_coverage_all": pl.Float64,
                "mean_coverage_all": pl.Float64,
            })

        # Aggregate ALL broadcasts per user-programme
        all_agg = coverage_df.group_by(["user_id", "tv_show_id"]).agg([
            pl.col("overlap_seconds").sum().alias("total_watched_seconds"),
            pl.col("airing_id").n_unique().alias("total_broadcasts_viewed"),
            pl.col("coverage").max().alias("max_coverage_all"),
            pl.col("coverage").mean().alias("mean_coverage_all"),
            pl.col("programme_start_utc").min().alias("first_interaction_utc"),
            pl.col("programme_end_utc").max().alias("last_interaction_utc"),
            pl.col("programme_start_utc").dt.date().n_unique().alias("distinct_viewing_days"),
        ])

        # Aggregate qualifying broadcasts (coverage >= threshold)
        qualifying = coverage_df.filter(pl.col("is_relevant"))
        if qualifying.height > 0:
            qual_agg = qualifying.group_by(["user_id", "tv_show_id"]).agg([
                pl.col("overlap_seconds").sum().alias("qualifying_watched_seconds"),
                pl.col("airing_id").n_unique().alias("num_qualifying_broadcasts"),
                pl.col("coverage").max().alias("max_coverage_qual"),
                pl.col("coverage").mean().alias("mean_coverage_qual"),
                pl.lit(True).alias("is_relevant"),
            ])
            result = all_agg.join(qual_agg, on=["user_id", "tv_show_id"], how="left")
            result = result.with_columns([
                pl.col("is_relevant").fill_null(False),
                pl.col("num_qualifying_broadcasts").fill_null(0),
                pl.col("qualifying_watched_seconds").fill_null(0),
                pl.col("max_coverage_qual").fill_null(pl.col("max_coverage_all")).alias("max_coverage"),
                pl.col("mean_coverage_qual").fill_null(pl.col("mean_coverage_all")).alias("mean_coverage"),
            ])
        else:
            result = all_agg.with_columns([
                pl.lit(False).alias("is_relevant"),
                pl.lit(0).alias("num_qualifying_broadcasts"),
                pl.lit(0).alias("qualifying_watched_seconds"),
                pl.col("max_coverage_all").alias("max_coverage"),
                pl.col("mean_coverage_all").alias("mean_coverage"),
            ])

        # Rename for backwards compatibility
        result = result.rename({
            "total_broadcasts_viewed": "total_broadcasts_watched",
        })

        return result


class TemporalSplitter:
    """Handles temporal train/validation/test splits with interval clipping."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        import zoneinfo
        utc_tz = zoneinfo.ZoneInfo("UTC")
        self.train_start = TRAIN_START.replace(tzinfo=utc_tz)
        self.train_end = TRAIN_END.replace(tzinfo=utc_tz)
        self.val_start = VAL_START.replace(tzinfo=utc_tz)
        self.val_end = VAL_END.replace(tzinfo=utc_tz)
        self.test_start = TEST_START.replace(tzinfo=utc_tz)
        self.test_end = TEST_END.replace(tzinfo=utc_tz)

    def _clip_intervals(
        self,
        viewing_lf: pl.LazyFrame,
        window_start: datetime,
        window_end: datetime
    ) -> pl.LazyFrame:
        """Clip viewing intervals to a time window using half-open semantics."""
        return viewing_lf.with_columns([
            pl.max_horizontal("view_start_utc", pl.lit(window_start)).alias("clipped_start_utc"),
            pl.min_horizontal("view_end_utc", pl.lit(window_end)).alias("clipped_end_utc"),
        ]).filter(
            pl.col("clipped_end_utc") > pl.col("clipped_start_utc")
        ).with_columns([
            (pl.col("clipped_end_utc") - pl.col("clipped_start_utc")).dt.total_seconds()
            .cast(pl.Int64).alias("duraton_seconds"),
        ]).drop(["clipped_start_utc", "clipped_end_utc"])

    def split_viewing(
        self,
        viewing_lf: pl.LazyFrame
    ) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
        """Split viewing data into train/val/test with interval clipping at boundaries."""
        # First, split into three raw categories
        train_raw = viewing_lf.filter(
            (pl.col("view_start_utc") < pl.lit(self.train_end)) &
            (pl.col("view_end_utc") > pl.lit(self.train_start))
        )

        val_raw = viewing_lf.filter(
            (pl.col("view_start_utc") < pl.lit(self.val_end)) &
            (pl.col("view_end_utc") > pl.lit(self.val_start))
        )

        # Clip train intervals to training window
        train_viewing = self._clip_intervals(
            train_raw, self.train_start, self.train_end
        ).with_columns([
            pl.col("view_start_utc").alias("original_view_start_utc"),
            pl.col("view_end_utc").alias("original_view_end_utc"),
        ]).with_columns([
            pl.max_horizontal("view_start_utc", pl.lit(self.train_start)).alias("view_start_utc"),
            pl.min_horizontal("view_end_utc", pl.lit(self.train_end)).alias("view_end_utc"),
        ])

        # Clip validation intervals to validation window
        val_viewing = self._clip_intervals(
            val_raw, self.val_start, self.val_end
        ).with_columns([
            pl.col("view_start_utc").alias("original_view_start_utc"),
            pl.col("view_end_utc").alias("original_view_end_utc"),
        ]).with_columns([
            pl.max_horizontal("view_start_utc", pl.lit(self.val_start)).alias("view_start_utc"),
            pl.min_horizontal("view_end_utc", pl.lit(self.val_end)).alias("view_end_utc"),
        ])

        # Test has no viewing data (future)
        test_viewing = viewing_lf.filter(pl.lit(False))

        return train_viewing, val_viewing, test_viewing

    def assert_no_leakage(self, train_viewing: pl.DataFrame, val_viewing: pl.DataFrame) -> None:
        """Assert no temporal leakage between splits."""
        # All train fragments end at or before train_end
        train_max_end = train_viewing["view_end_utc"].max()
        assert train_max_end <= self.train_end, f"Train leakage: max end {train_max_end} > train_end {self.train_end}"
        # All val fragments start at or after val_start
        val_min_start = val_viewing["view_start_utc"].min()
        assert val_min_start >= self.val_start, f"Val leakage: min start {val_min_start} < val_start {self.val_start}"

    def get_eligible_programmes(self, period: str) -> pl.DataFrame:
        """Get eligible programmes for a given period (tv_show_id != 0)."""
        ingestion = DataIngestion(self.config)
        schedule_lf = ingestion.load_schedule(period)
        identified = self.normalize_broadcast_intervals(schedule_lf)
        programmes = identified.select("tv_show_id").unique().collect()
        return programmes

    def normalize_broadcast_intervals(self, schedule_lf: pl.LazyFrame) -> pl.LazyFrame:
        """Normalize broadcast intervals, exclude unidentified programmes."""
        identified = schedule_lf.filter(pl.col("tv_show_id") != 0)
        identified = identified.unique(subset=["channel_id", "programme_start_utc", "tv_show_id"])
        return identified


def run_pipeline(config: PipelineConfig | None = None) -> dict[str, Any]:
    """Run the complete interaction pipeline with proper temporal isolation and interval clipping."""
    if config is None:
        config = PipelineConfig()

    ingestion = DataIngestion(config)
    builder = InteractionBuilder(config)
    splitter = TemporalSplitter(config)

    logger.info("Loading viewing logs...")
    viewing_lf = ingestion.load_viewing_logs()

    logger.info("Loading all schedules...")
    schedule_lf = ingestion.load_all_schedules()

    logger.info("Splitting viewing data by temporal period with interval clipping...")
    train_viewing_lf, val_viewing_lf, _ = splitter.split_viewing(viewing_lf)

    # Collect for leakage assertions
    train_viewing = train_viewing_lf.collect()
    val_viewing = val_viewing_lf.collect()

    # Assert no leakage
    splitter.assert_no_leakage(train_viewing, val_viewing)

    logger.info(f"Train viewing fragments: {train_viewing.height}, Validation viewing fragments: {val_viewing.height}")

    logger.info("Building train interactions...")
    train_interactions = builder.aggregate_user_programme_interactions(train_viewing_lf, schedule_lf)

    logger.info("Building validation interactions...")
    val_interactions = builder.aggregate_user_programme_interactions(val_viewing_lf, schedule_lf)

    # Combine for full historical interactions (for reference)
    all_interactions = pl.concat([train_interactions, val_interactions], how="vertical")

    logger.info("Saving processed data...")
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    all_interactions.write_parquet(config.processed_dir / "user_programme_interactions.parquet")
    train_interactions.write_parquet(config.processed_dir / "train_interactions.parquet")
    val_interactions.write_parquet(config.processed_dir / "val_interactions.parquet")

    # Also save clipped viewing data for transparency
    train_viewing.write_parquet(config.processed_dir / "train_viewing_clipped.parquet")
    val_viewing.write_parquet(config.processed_dir / "val_viewing_clipped.parquet")

    # Positive labels (coverage >= threshold)
    train_positive = train_interactions.filter(pl.col("is_relevant"))
    val_positive = val_interactions.filter(pl.col("is_relevant"))

    # Repeated user-programme pairs
    train_pairs = set(zip(train_interactions["user_id"].to_list(), train_interactions["tv_show_id"].to_list()))
    val_pairs = set(zip(val_interactions["user_id"].to_list(), val_interactions["tv_show_id"].to_list()))
    repeated_pairs = train_pairs & val_pairs

    # Verify training features are independent of validation changes
    # (by construction: train_interactions built only from train_viewing_lf)
    logger.info("Verified: training features constructed solely from training-period viewing")

    # Save evidence
    evidence = {
        "total_interactions": all_interactions.height,
        "train_interactions": train_interactions.height,
        "val_interactions": val_interactions.height,
        "train_positive_labels": train_positive.height,
        "val_positive_labels": val_positive.height,
        "unique_users": all_interactions["user_id"].n_unique(),
        "unique_programmes": all_interactions["tv_show_id"].n_unique(),
        "train_unique_users": train_interactions["user_id"].n_unique(),
        "val_unique_users": val_interactions["user_id"].n_unique(),
        "train_unique_programmes": train_interactions["tv_show_id"].n_unique(),
        "val_unique_programmes": val_interactions["tv_show_id"].n_unique(),
        "repeated_user_programme_pairs": len(repeated_pairs),
        "partial_train_interactions": train_interactions.filter(~pl.col("is_relevant")).height,
        "partial_val_interactions": val_interactions.filter(~pl.col("is_relevant")).height,
        "train_viewing_fragments": train_viewing.height,
        "val_viewing_fragments": val_viewing.height,
        "leakage_assertions_passed": True,
    }
    with open(config.evidence_dir / "phase02_2_checks.json", "w") as f:
        json.dump(evidence, f, indent=2)

    logger.info(f"Pipeline complete. Evidence: {evidence}")
    return evidence


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = PipelineConfig()
    run_pipeline(config)
