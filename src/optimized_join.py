"""
Optimized interval join using DuckDB for efficient temporal overlaps.
"""
import duckdb
import polars as pl
from datetime import datetime
from pathlib import Path

from .constants import KYIV_TZ, COVERAGE_THRESHOLD, PipelineConfig


class OptimizedInteractionBuilder:
    """Uses DuckDB for efficient interval joins."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.conn = duckdb.connect(":memory:")
        self.conn.execute("PRAGMA threads=8")
        self.conn.execute("PRAGMA memory_limit='8GB'")
        self.conn.execute("SET TimeZone='UTC'")

    def register_viewing(self, viewing_lf: pl.LazyFrame) -> None:
        """Register viewing data in DuckDB."""
        viewing_df = viewing_lf.collect()
        # Convert to pandas for DuckDB registration
        viewing_pd = viewing_df.to_pandas()
        self.conn.register("viewing", viewing_pd)

    def register_schedule(self, schedule_lf: pl.LazyFrame) -> None:
        """Register schedule data in DuckDB."""
        schedule_df = schedule_lf.collect()
        schedule_pd = schedule_df.to_pandas()
        self.conn.register("schedule", schedule_pd)

    def compute_intersections(self) -> pl.DataFrame:
        """Compute viewing-schedule intersections using DuckDB range join."""
        query = """
        SELECT 
            v.user_id,
            v.channel_id,
            v.view_start_utc,
            v.view_end_utc,
            v.duraton_seconds,
            s.airing_id,
            s.programme_start_utc,
            s.programme_end_utc,
            s.tv_show_id,
            s.programme_duration_seconds,
            GREATEST(v.view_start_utc, s.programme_start_utc) AS overlap_start_utc,
            LEAST(v.view_end_utc, s.programme_end_utc) AS overlap_end_utc,
            EXTRACT(EPOCH FROM (LEAST(v.view_end_utc, s.programme_end_utc) - GREATEST(v.view_start_utc, s.programme_start_utc)))::BIGINT AS overlap_seconds
        FROM viewing v
        INNER JOIN schedule s
            ON v.channel_id = s.channel_id
            AND v.view_start_utc < s.programme_end_utc
            AND v.view_end_utc > s.programme_start_utc
        """

        result = self.conn.execute(query).fetchdf()
        # Ensure UTC timezone for all datetime columns
        df = pl.from_pandas(result)
        for col in df.columns:
            if df[col].dtype == pl.Datetime:
                if df[col].dtype.time_zone is not None:
                    df = df.with_columns(pl.col(col).dt.convert_time_zone("UTC"))
                else:
                    df = df.with_columns(pl.col(col).dt.replace_time_zone("UTC"))
        return df

    def close(self):
        self.conn.close()


def compute_intersections_optimized(viewing_lf: pl.LazyFrame, schedule_lf: pl.LazyFrame) -> pl.DataFrame:
    """Compute intersections using optimized DuckDB join."""
    builder = OptimizedInteractionBuilder(PipelineConfig())
    try:
        builder.register_viewing(viewing_lf)
        builder.register_schedule(schedule_lf)
        intersections = builder.compute_intersections()
        return intersections
    finally:
        builder.close()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    from src.interaction_pipeline import DataIngestion, InteractionBuilder

    config = PipelineConfig()
    ingestion = DataIngestion(config)
    builder = InteractionBuilder(config)

    viewing = ingestion.load_viewing_logs().head(50000)
    schedule = ingestion.load_schedule("11-20").head(50000)

    viewing_norm = builder.normalize_viewing_intervals(viewing)
    schedule_norm = builder.normalize_broadcast_intervals(schedule)

    print("Computing intersections with DuckDB...")
    intersections = compute_intersections_optimized(viewing_norm, schedule_norm)
    print(f"Intersections: {intersections.height}")

    # Merge fragments
    merged = builder._merge_fragments_eager(intersections)
    print(f"Merged: {merged.height}")

    # Coverage
    coverage = builder._calculate_coverage_eager(merged)
    print(f"Coverage calculated: {coverage.height}")
    print(coverage.head())
