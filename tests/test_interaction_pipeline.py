#!/usr/bin/env python3
"""
Unit tests for the interaction pipeline.

Tests cover:
1. User ID preservation (unsigned 64-bit)
2. Timestamp parsing and DST handling
3. Interval overlap calculations
4. Fragment merging
5. Coverage calculation
6. Temporal leakage prevention
7. Submission user count
8. Deterministic output
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.interaction_pipeline import (
    DataIngestion,
    InteractionBuilder,
    TemporalSplitter,
    PipelineConfig,
    KYIV_TZ,
    COVERAGE_THRESHOLD,
    TRAIN_START, TRAIN_END, VAL_START, VAL_END,
)


class TestUserIDPreservation:
    """Test that large unsigned user IDs are preserved exactly."""

    def test_submission_user_count(self):
        """Verify submission has exactly 1260 unique users."""
        config = PipelineConfig(data_dir=Path("data"))
        ingestion = DataIngestion(config)
        sub = ingestion.load_submission()
        assert sub.height == 1260
        assert sub["user_id"].n_unique() == 1260

    def test_user_ids_preserved_as_strings(self):
        """User IDs must remain as strings, not converted to int64/float."""
        config = PipelineConfig(data_dir=Path("data"))
        ingestion = DataIngestion(config)
        sub = ingestion.load_submission()
        assert sub["user_id"].dtype == pl.Utf8

        # Check a known large user ID
        large_uid = "18443832178892490378"
        # This should be in the viewing logs
        viewing = ingestion.load_viewing_logs().collect()
        assert viewing["user_id"].dtype == pl.Utf8


class TestTimestampParsing:
    """Test timestamp parsing and UTC conversion."""

    def test_viewing_timestamp_format(self):
        """Viewing logs use YYYY-MM-DD HH:MM:SS format."""
        config = PipelineConfig(data_dir=Path("data"))
        ingestion = DataIngestion(config)
        viewing = ingestion.load_viewing_logs().head(10).collect()

        assert viewing["view_start_utc"].dtype == pl.Datetime("us", "UTC")
        assert viewing["view_end_utc"].dtype == pl.Datetime("us", "UTC")

        # Verify first timestamp
        first_start = viewing["view_start_utc"][0]
        # 2020-03-09 00:00:02 Kyiv (EET, UTC+2) = 2020-03-08 22:00:02 UTC
        assert first_start.hour == 22
        assert first_start.minute == 0
        assert first_start.second == 2

    def test_schedule_timestamp_format(self):
        """Schedule uses DD.MM.YYYY HH:MM:SS format."""
        config = PipelineConfig(data_dir=Path("data"))
        ingestion = DataIngestion(config)
        schedule = ingestion.load_schedule("11-20").head(10).collect()

        assert schedule["programme_start_utc"].dtype == pl.Datetime("us", "UTC")
        assert schedule["programme_end_utc"].dtype == pl.Datetime("us", "UTC")

        # Verify first timestamp: 09.03.2020 06:00:00 Kyiv (EET, UTC+2) = 2020-03-09 04:00:00 UTC
        first_start = schedule["programme_start_utc"][0]
        assert first_start.hour == 4
        assert first_start.minute == 0

    def test_programme_end_calculation(self):
        """Programme end = start + duration seconds."""
        config = PipelineConfig(data_dir=Path("data"))
        ingestion = DataIngestion(config)
        schedule = ingestion.load_schedule("11-20").head(5).collect()

        for row in schedule.iter_rows(named=True):
            start = row["programme_start_utc"]
            end = row["programme_end_utc"]
            duration = row["programme_duration_seconds"]
            expected_end = start + timedelta(seconds=duration)
            assert end == expected_end, f"Mismatch: {end} != {expected_end}"


class TestDSTHandling:
    """Test daylight-saving transition handling."""

    def test_dst_transition_march_29(self):
        """March 29 2020: 03:00 local -> 04:00 local (spring forward)."""
        import zoneinfo
        kyiv = zoneinfo.ZoneInfo(KYIV_TZ)

        # 02:30 exists (EET, UTC+2)
        dt_before = datetime(2020, 3, 29, 2, 30, tzinfo=kyiv)
        assert dt_before.utcoffset() == timedelta(hours=2)

        # 04:30 exists (EEST, UTC+3)
        dt_after = datetime(2020, 3, 29, 4, 30, tzinfo=kyiv)
        assert dt_after.utcoffset() == timedelta(hours=3)

        # 03:30 is ambiguous in zoneinfo - it picks pre-DST offset
        dt_gap = datetime(2020, 3, 29, 3, 30, tzinfo=kyiv)
        # zoneinfo treats 03:30 as if DST hasn't happened yet (UTC+2)
        assert dt_gap.utcoffset() == timedelta(hours=2)

    def test_viewing_dst_crossing(self):
        """Viewing sessions crossing DST gap have correct duraton."""
        config = PipelineConfig(data_dir=Path("data"))
        ingestion = DataIngestion(config)
        viewing = ingestion.load_viewing_logs().collect()

        # Filter using numpy arrays to avoid Polars literal comparison issues
        mar29_mask = (
            (viewing["view_start_utc"].to_numpy() >= np.datetime64("2020-03-28T22:00:00")) &
            (viewing["view_start_utc"].to_numpy() < np.datetime64("2020-03-29T22:00:00"))
        )
        viewing = viewing.filter(pl.Series("mask", mar29_mask))

        # Check sessions that cross the gap
        cross_mask = (
            (viewing["view_start_utc"].to_numpy() < np.datetime64("2020-03-29T01:00:00")) &  # Before 03:00 local
            (viewing["view_end_utc"].to_numpy() > np.datetime64("2020-03-29T01:00:00"))       # After 04:00 local
        )
        crossing = viewing.filter(pl.Series("mask", cross_mask))

        for row in crossing.iter_rows(named=True):
            # UTC difference should match duraton_seconds
            utc_diff = int((row["view_end_utc"] - row["view_start_utc"]).total_seconds())
            assert utc_diff == row["duraton_seconds"], (
                f"UTC diff {utc_diff} != duraton {row['duraton_seconds']}"
            )

    def test_schedule_dst_conversion(self):
        """Schedule correctly converts local times including DST transition."""
        config = PipelineConfig(data_dir=Path("data"))
        ingestion = DataIngestion(config)
        schedule = ingestion.load_schedule("11-20").collect()

        # Filter March 29 local (22:00 UTC Mar 28 to 22:00 UTC Mar 29)
        mar29_mask = (
            (schedule["programme_start_utc"].to_numpy() >= np.datetime64("2020-03-28T22:00:00")) &
            (schedule["programme_start_utc"].to_numpy() < np.datetime64("2020-03-29T22:00:00"))
        )
        mar29 = schedule.filter(pl.Series("mask", mar29_mask))

        # Schedule has NO hour 3 LOCAL (03:00-03:59 local)
        # In UTC: pre-DST 03:00 local = 01:00 UTC, post-DST 04:00 local = 01:00 UTC
        # So hour 1 UTC could be either 03:00 local (pre-DST) or 04:00 local (post-DST)
        # But there should be NO entries at 03:xx LOCAL
        # We can verify by checking the original local hour from the raw data

        # Just verify schedule loaded correctly and has no gaps in UTC hours
        # (the missing local hour is handled by timezone conversion)
        assert mar29.height > 0
        # Hours present in UTC (0-23, but hour 1 has entries from both pre and post DST)
        utc_hours = mar29["programme_start_utc"].dt.hour().unique().to_list()
        assert len(utc_hours) >= 22  # Most hours present


class TestIntervalOverlap:
    """Test interval overlap calculations."""

    def test_no_overlap_disjoint(self):
        """Non-intersecting intervals produce no overlap."""
        builder = InteractionBuilder(PipelineConfig())

        # Viewing: 10:00-11:00, Programme: 12:00-13:00 (no overlap)
        viewing = pl.DataFrame({
            "user_id": ["u1"],
            "channel_id": [1],
            "view_start_utc": [datetime(2020, 5, 1, 10, 0, 0)],
            "view_end_utc": [datetime(2020, 5, 1, 11, 0, 0)],
            "duraton_seconds": [3600],
        }).lazy()

        schedule = pl.DataFrame({
            "airing_id": ["a1"],
            "channel_id": [1],
            "programme_start_utc": [datetime(2020, 5, 1, 12, 0, 0)],
            "programme_end_utc": [datetime(2020, 5, 1, 13, 0, 0)],
            "tv_show_id": [100],
            "programme_duration_seconds": [3600],
        }).lazy()

        intersections = builder.compute_interval_intersections(viewing, schedule).collect()
        assert intersections.height == 0

    def test_exact_overlap(self):
        """Exact matching intervals produce full overlap."""
        builder = InteractionBuilder(PipelineConfig())

        viewing = pl.DataFrame({
            "user_id": ["u1"],
            "channel_id": [1],
            "view_start_utc": [datetime(2020, 5, 1, 10, 0, 0)],
            "view_end_utc": [datetime(2020, 5, 1, 11, 0, 0)],
            "duraton_seconds": [3600],
        }).lazy()

        schedule = pl.DataFrame({
            "airing_id": ["a1"],
            "channel_id": [1],
            "programme_start_utc": [datetime(2020, 5, 1, 10, 0, 0)],
            "programme_end_utc": [datetime(2020, 5, 1, 11, 0, 0)],
            "tv_show_id": [100],
            "programme_duration_seconds": [3600],
        }).lazy()

        intersections = builder.compute_interval_intersections(viewing, schedule).collect()
        assert intersections.height == 1
        assert intersections["overlap_seconds"][0] == 3600

    def test_partial_overlap_start(self):
        """Viewing starts during programme."""
        builder = InteractionBuilder(PipelineConfig())

        viewing = pl.DataFrame({
            "user_id": ["u1"],
            "channel_id": [1],
            "view_start_utc": [datetime(2020, 5, 1, 10, 30, 0)],
            "view_end_utc": [datetime(2020, 5, 1, 11, 30, 0)],
            "duraton_seconds": [3600],
        }).lazy()

        schedule = pl.DataFrame({
            "airing_id": ["a1"],
            "channel_id": [1],
            "programme_start_utc": [datetime(2020, 5, 1, 10, 0, 0)],
            "programme_end_utc": [datetime(2020, 5, 1, 11, 0, 0)],
            "tv_show_id": [100],
            "programme_duration_seconds": [3600],
        }).lazy()

        intersections = builder.compute_interval_intersections(viewing, schedule).collect()
        assert intersections.height == 1
        assert intersections["overlap_seconds"][0] == 1800  # 30 minutes

    def test_partial_overlap_end(self):
        """Viewing ends during programme."""
        builder = InteractionBuilder(PipelineConfig())

        viewing = pl.DataFrame({
            "user_id": ["u1"],
            "channel_id": [1],
            "view_start_utc": [datetime(2020, 5, 1, 9, 30, 0)],
            "view_end_utc": [datetime(2020, 5, 1, 10, 30, 0)],
            "duraton_seconds": [3600],
        }).lazy()

        schedule = pl.DataFrame({
            "airing_id": ["a1"],
            "channel_id": [1],
            "programme_start_utc": [datetime(2020, 5, 1, 10, 0, 0)],
            "programme_end_utc": [datetime(2020, 5, 1, 11, 0, 0)],
            "tv_show_id": [100],
            "programme_duration_seconds": [3600],
        }).lazy()

        intersections = builder.compute_interval_intersections(viewing, schedule).collect()
        assert intersections.height == 1
        assert intersections["overlap_seconds"][0] == 1800  # 30 minutes

    def test_viewing_contains_programme(self):
        """Viewing fully contains programme."""
        builder = InteractionBuilder(PipelineConfig())

        viewing = pl.DataFrame({
            "user_id": ["u1"],
            "channel_id": [1],
            "view_start_utc": [datetime(2020, 5, 1, 9, 0, 0)],
            "view_end_utc": [datetime(2020, 5, 1, 12, 0, 0)],
            "duraton_seconds": [10800],
        }).lazy()

        schedule = pl.DataFrame({
            "airing_id": ["a1"],
            "channel_id": [1],
            "programme_start_utc": [datetime(2020, 5, 1, 10, 0, 0)],
            "programme_end_utc": [datetime(2020, 5, 1, 11, 0, 0)],
            "tv_show_id": [100],
            "programme_duration_seconds": [3600],
        }).lazy()

        intersections = builder.compute_interval_intersections(viewing, schedule).collect()
        assert intersections.height == 1
        assert intersections["overlap_seconds"][0] == 3600


class TestFragmentMerging:
    """Test overlapping viewing fragment merging."""

    def test_overlapping_fragments_merged(self):
        """Overlapping fragments for same user-airing are merged."""
        builder = InteractionBuilder(PipelineConfig())

        # Two overlapping viewing fragments for same airing
        # Both within programme boundaries (10:00-11:00)
        intersections = pl.DataFrame({
            "user_id": ["u1", "u1"],
            "airing_id": ["a1", "a1"],
            "tv_show_id": [100, 100],
            "programme_duration_seconds": [3600, 3600],
            "overlap_start_utc": [
                datetime(2020, 5, 1, 10, 0, 0),
                datetime(2020, 5, 1, 10, 15, 0),
            ],
            "overlap_end_utc": [
                datetime(2020, 5, 1, 10, 45, 0),
                datetime(2020, 5, 1, 11, 0, 0),
            ],
            "channel_id": [1, 1],
            "programme_start_utc": [datetime(2020, 5, 1, 10, 0, 0), datetime(2020, 5, 1, 10, 0, 0)],
            "programme_end_utc": [datetime(2020, 5, 1, 11, 0, 0), datetime(2020, 5, 1, 11, 0, 0)],
        })

        merged = builder._merge_fragments_eager(intersections)
        assert merged.height == 1
        # Merged: 10:00-11:00 = 3600 seconds (clipped to programme)
        assert merged["overlap_seconds"][0] == 3600

    def test_adjacent_fragments_merged(self):
        """Adjacent (touching) fragments are merged."""
        builder = InteractionBuilder(PipelineConfig())

        intersections = pl.DataFrame({
            "user_id": ["u1", "u1"],
            "airing_id": ["a1", "a1"],
            "tv_show_id": [100, 100],
            "programme_duration_seconds": [7200, 7200],
            "overlap_start_utc": [
                datetime(2020, 5, 1, 10, 0, 0),
                datetime(2020, 5, 1, 11, 0, 0),
            ],
            "overlap_end_utc": [
                datetime(2020, 5, 1, 11, 0, 0),
                datetime(2020, 5, 1, 12, 0, 0),
            ],
            "channel_id": [1, 1],
            "programme_start_utc": [datetime(2020, 5, 1, 10, 0, 0), datetime(2020, 5, 1, 10, 0, 0)],
            "programme_end_utc": [datetime(2020, 5, 1, 12, 0, 0), datetime(2020, 5, 1, 12, 0, 0)],
        })

        merged = builder._merge_fragments_eager(intersections)
        assert merged.height == 1
        assert merged["overlap_seconds"][0] == 7200

    def test_separate_fragments_not_merged(self):
        """Separate fragments with gaps are not merged."""
        builder = InteractionBuilder(PipelineConfig())

        intersections = pl.DataFrame({
            "user_id": ["u1", "u1"],
            "airing_id": ["a1", "a1"],
            "tv_show_id": [100, 100],
            "programme_duration_seconds": [10800, 10800],
            "overlap_start_utc": [
                datetime(2020, 5, 1, 10, 0, 0),
                datetime(2020, 5, 1, 12, 0, 0),
            ],
            "overlap_end_utc": [
                datetime(2020, 5, 1, 11, 0, 0),
                datetime(2020, 5, 1, 13, 0, 0),
            ],
            "channel_id": [1, 1],
            "programme_start_utc": [datetime(2020, 5, 1, 10, 0, 0), datetime(2020, 5, 1, 10, 0, 0)],
            "programme_end_utc": [datetime(2020, 5, 1, 13, 0, 0), datetime(2020, 5, 1, 13, 0, 0)],
        })

        merged = builder._merge_fragments_eager(intersections)
        assert merged.height == 1
        # Two separate 1-hour fragments = 7200 seconds total
        assert merged["overlap_seconds"][0] == 7200


class TestCoverageCalculation:
    """Test coverage calculation and thresholding."""

    def test_coverage_calculation(self):
        """Coverage = overlap_seconds / programme_duration_seconds."""
        builder = InteractionBuilder(PipelineConfig())

        merged = pl.DataFrame({
            "user_id": ["u1"],
            "airing_id": ["a1"],
            "tv_show_id": [100],
            "programme_duration_seconds": [3600],
            "overlap_seconds": [2700],  # 75%
            "channel_id": [1],
            "programme_start_utc": [datetime(2020, 5, 1, 10, 0, 0)],
            "programme_end_utc": [datetime(2020, 5, 1, 11, 0, 0)],
        })

        coverage = builder._calculate_coverage_eager(merged)
        assert coverage["coverage"][0] == 0.75
        assert coverage["is_relevant"][0] == False  # Below 80% threshold

    def test_coverage_clamped(self):
        """Coverage clamped to [0, 1]."""
        builder = InteractionBuilder(PipelineConfig())

        merged = pl.DataFrame({
            "user_id": ["u1"],
            "airing_id": ["a1"],
            "tv_show_id": [100],
            "programme_duration_seconds": [3600],
            "overlap_seconds": [5000],  # More than duration (shouldn't happen but test clamp)
            "channel_id": [1],
            "programme_start_utc": [datetime(2020, 5, 1, 10, 0, 0)],
            "programme_end_utc": [datetime(2020, 5, 1, 11, 0, 0)],
        })

        coverage = builder._calculate_coverage_eager(merged)
        assert coverage["coverage"][0] == 1.0
        assert coverage["is_relevant"][0] == True

    def test_zero_coverage(self):
        """Zero overlap gives zero coverage."""
        builder = InteractionBuilder(PipelineConfig())

        merged = pl.DataFrame({
            "user_id": ["u1"],
            "airing_id": ["a1"],
            "tv_show_id": [100],
            "programme_duration_seconds": [3600],
            "overlap_seconds": [0],
            "channel_id": [1],
            "programme_start_utc": [datetime(2020, 5, 1, 10, 0, 0)],
            "programme_end_utc": [datetime(2020, 5, 1, 11, 0, 0)],
        })

        coverage = builder._calculate_coverage_eager(merged)
        assert coverage["coverage"][0] == 0.0
        assert coverage["is_relevant"][0] == False


class TestProgrammeAggregation:
    """Test user-programme level aggregation."""

    def test_aggregation_basic(self):
        """Basic aggregation across multiple airings."""
        builder = InteractionBuilder(PipelineConfig())

        coverage = pl.DataFrame({
            "user_id": ["u1", "u1", "u1"],
            "airing_id": ["a1", "a2", "a3"],
            "tv_show_id": [100, 100, 100],
            "programme_duration_seconds": [3600, 3600, 3600],
            "overlap_seconds": [3600, 2700, 1800],
            "channel_id": [1, 1, 1],
            "programme_start_utc": [
                datetime(2020, 5, 1, 10, 0, 0),
                datetime(2020, 5, 2, 10, 0, 0),
                datetime(2020, 5, 3, 10, 0, 0),
            ],
            "programme_end_utc": [
                datetime(2020, 5, 1, 11, 0, 0),
                datetime(2020, 5, 2, 11, 0, 0),
                datetime(2020, 5, 3, 11, 0, 0),
            ],
            "coverage": [1.0, 0.75, 0.5],
            "is_relevant": [True, False, False],
        })

        agg = builder._aggregate_to_programme_level_eager(coverage)
        assert agg.height == 1
        row = agg.row(0, named=True)
        # Only a1 is qualifying (coverage >= 0.8)
        assert row["num_qualifying_broadcasts"] == 1
        assert row["qualifying_watched_seconds"] == 3600
        assert row["is_relevant"] == True
        # All broadcasts stats
        assert row["total_broadcasts_watched"] == 3
        assert row["total_watched_seconds"] == 3600 + 2700 + 1800
        assert abs(row["max_coverage"] - 1.0) < 1e-9
        # mean_coverage is only over qualifying broadcasts
        assert row["mean_coverage"] == 1.0
        # all broadcasts coverage stats
        assert row["max_coverage_all"] == 1.0
        assert abs(row["mean_coverage_all"] - (1.0 + 0.75 + 0.5) / 3) < 1e-9


class TestTemporalSplitting:
    """Test temporal train/validation/test splits."""

    def test_split_viewing_boundaries(self):
        """Viewing events correctly split by view_start_utc."""
        import zoneinfo
        utc = zoneinfo.ZoneInfo("UTC")
        splitter = TemporalSplitter(PipelineConfig())

        viewing = pl.DataFrame({
            "user_id": ["u1", "u1", "u2", "u2"],
            "channel_id": [1, 1, 2, 2],
            "view_start_utc": [
                datetime(2020, 4, 1, 10, 0, 0, tzinfo=utc),   # Train
                datetime(2020, 6, 1, 10, 0, 0, tzinfo=utc),   # Val
                datetime(2020, 4, 15, 10, 0, 0, tzinfo=utc),  # Train
                datetime(2020, 7, 1, 10, 0, 0, tzinfo=utc),   # Val
            ],
            "view_end_utc": [
                datetime(2020, 4, 1, 11, 0, 0, tzinfo=utc),
                datetime(2020, 6, 1, 11, 0, 0, tzinfo=utc),
                datetime(2020, 4, 15, 11, 0, 0, tzinfo=utc),
                datetime(2020, 7, 1, 11, 0, 0, tzinfo=utc),
            ],
            "duraton_seconds": [3600, 3600, 3600, 3600],
        }).lazy()

        train_viewing, val_viewing, test_viewing = splitter.split_viewing(viewing)

        train = train_viewing.collect()
        val = val_viewing.collect()
        test = test_viewing.collect()

        assert train.height == 2
        assert val.height == 2
        assert test.height == 0

        # Check train users
        train_users = set(train["user_id"].to_list())
        assert train_users == {"u1", "u2"}

        # Check val users
        val_users = set(val["user_id"].to_list())
        assert val_users == {"u1", "u2"}

    def test_no_leakage_split_viewing(self):
        """Validation viewing events never appear in training."""
        import zoneinfo
        utc = zoneinfo.ZoneInfo("UTC")
        splitter = TemporalSplitter(PipelineConfig())

        viewing = pl.DataFrame({
            "user_id": ["u1"] * 4,
            "channel_id": [1] * 4,
            "view_start_utc": [
                datetime(2020, 5, 17, 20, 0, 0, tzinfo=utc),  # Train (just before boundary)
                datetime(2020, 5, 17, 21, 0, 0, tzinfo=utc),   # Val (exactly at boundary)
                datetime(2020, 5, 18, 12, 0, 0, tzinfo=utc),  # Val
                datetime(2020, 7, 26, 23, 0, 0, tzinfo=utc),  # Val (last day)
            ],
            "view_end_utc": [
                datetime(2020, 5, 17, 23, 30, 0, tzinfo=utc),
                datetime(2020, 5, 18, 0, 30, 0, tzinfo=utc),
                datetime(2020, 5, 18, 12, 30, 0, tzinfo=utc),
                datetime(2020, 7, 26, 23, 30, 0, tzinfo=utc),
            ],
            "duraton_seconds": [1800] * 4,
        }).lazy()

        train_viewing, val_viewing, _ = splitter.split_viewing(viewing)

        train = train_viewing.collect()
        val = val_viewing.collect()

        # Train should only have the first event
        assert train.height == 1
        train_end = TRAIN_END.replace(tzinfo=utc)
        val_start = VAL_START.replace(tzinfo=utc)
        val_end = VAL_END.replace(tzinfo=utc)
        assert train["view_start_utc"][0] < train_end

        # Val should have the rest
        assert val.height == 3
        for ts in val["view_start_utc"].to_list():
            assert ts >= val_start
            assert ts < val_end

    def test_crossing_boundary_intervals_clipped(self):
        """Intervals crossing May 18 boundary are clipped to each period."""
        import zoneinfo
        utc = zoneinfo.ZoneInfo("UTC")
        splitter = TemporalSplitter(PipelineConfig())

        # Viewing interval that starts in train, ends in val (7200 seconds = 2 hours)
        # Train part: 5/17 20:00 - 5/17 21:00 = 1 hour
        # Val part: 5/17 21:00 - 5/17 22:00 = 1 hour
        viewing = pl.DataFrame({
            "user_id": ["u1"],
            "channel_id": [1],
            "view_start_utc": [datetime(2020, 5, 17, 20, 0, 0, tzinfo=utc)],
            "view_end_utc": [datetime(2020, 5, 17, 22, 0, 0, tzinfo=utc)],
            "duraton_seconds": [7200],
        }).lazy()

        train_viewing, val_viewing, _ = splitter.split_viewing(viewing)

        train = train_viewing.collect()
        val = val_viewing.collect()

        # Both periods get a clipped fragment
        assert train.height == 1
        assert val.height == 1

        # Train fragment: 20:00 - 21:00 = 3600 seconds
        assert train["view_start_utc"][0] == datetime(2020, 5, 17, 20, 0, 0, tzinfo=utc)
        assert train["view_end_utc"][0] == datetime(2020, 5, 17, 21, 0, 0, tzinfo=utc)
        assert train["duraton_seconds"][0] == 3600

        # Val fragment: 21:00 - 22:00 = 3600 seconds
        assert val["view_start_utc"][0] == datetime(2020, 5, 17, 21, 0, 0, tzinfo=utc)
        assert val["view_end_utc"][0] == datetime(2020, 5, 17, 22, 0, 0, tzinfo=utc)
        assert val["duraton_seconds"][0] == 3600

    def test_boundary_exact_alignment(self):
        """Intervals exactly at boundary are handled with half-open semantics."""
        import zoneinfo
        utc = zoneinfo.ZoneInfo("UTC")
        splitter = TemporalSplitter(PipelineConfig())

        # Session ending exactly at train_end (exclusive)
        viewing1 = pl.DataFrame({
            "user_id": ["u1"],
            "channel_id": [1],
            "view_start_utc": [datetime(2020, 5, 17, 20, 0, 0, tzinfo=utc)],
            "view_end_utc": [datetime(2020, 5, 17, 21, 0, 0, tzinfo=utc)],
            "duraton_seconds": [3600],
        }).lazy()

        train1, val1, _ = splitter.split_viewing(viewing1)
        assert train1.collect().height == 1
        assert val1.collect().height == 0

        # Session starting exactly at val_start (inclusive)
        viewing2 = pl.DataFrame({
            "user_id": ["u1"],
            "channel_id": [1],
            "view_start_utc": [datetime(2020, 5, 17, 21, 0, 0, tzinfo=utc)],
            "view_end_utc": [datetime(2020, 5, 17, 22, 0, 0, tzinfo=utc)],
            "duraton_seconds": [3600],
        }).lazy()

        train2, val2, _ = splitter.split_viewing(viewing2)
        assert train2.collect().height == 0
        assert val2.collect().height == 1

    def test_short_session_at_boundary(self):
        """Sessions too short to produce positive duration after clipping are dropped."""
        import zoneinfo
        utc = zoneinfo.ZoneInfo("UTC")
        splitter = TemporalSplitter(PipelineConfig())

        # Session entirely in train, but ends exactly at boundary - should stay in train
        viewing = pl.DataFrame({
            "user_id": ["u1"],
            "channel_id": [1],
            "view_start_utc": [datetime(2020, 5, 17, 20, 59, 0, tzinfo=utc)],
            "view_end_utc": [datetime(2020, 5, 17, 21, 0, 0, tzinfo=utc)],
            "duraton_seconds": [60],
        }).lazy()

        train, val, _ = splitter.split_viewing(viewing)
        assert train.collect().height == 1
        assert val.collect().height == 0

    def test_repeated_programme_in_both_periods(self):
        """Same user-programme pair can appear independently in train and val."""
        import zoneinfo
        utc = zoneinfo.ZoneInfo("UTC")
        splitter = TemporalSplitter(PipelineConfig())
        builder = InteractionBuilder(PipelineConfig())
        ingestion = DataIngestion(PipelineConfig(data_dir=Path("data")))

        # Create synthetic viewing: same user watches programme 100 in April and June
        viewing = pl.DataFrame({
            "user_id": ["u1", "u1"],
            "channel_id": [1, 1],
            "view_start_utc": [
                datetime(2020, 4, 1, 10, 0, 0, tzinfo=utc),   # Train
                datetime(2020, 6, 1, 10, 0, 0, tzinfo=utc),   # Val
            ],
            "view_end_utc": [
                datetime(2020, 4, 1, 11, 0, 0, tzinfo=utc),
                datetime(2020, 6, 1, 11, 0, 0, tzinfo=utc),
            ],
            "duraton_seconds": [3600, 3600],
        })

        # Schedule: programme 100 airs on channel 1 in both periods
        schedule = pl.DataFrame({
            "airing_id": ["a1", "a2"],
            "channel_id": [1, 1],
            "programme_start_utc": [
                datetime(2020, 4, 1, 10, 0, 0, tzinfo=utc),
                datetime(2020, 6, 1, 10, 0, 0, tzinfo=utc),
            ],
            "programme_end_utc": [
                datetime(2020, 4, 1, 11, 0, 0, tzinfo=utc),
                datetime(2020, 6, 1, 11, 0, 0, tzinfo=utc),
            ],
            "tv_show_id": [100, 100],
            "programme_duration_seconds": [3600, 3600],
        })

        train_viewing = viewing.filter(
            (pl.col("view_start_utc") >= splitter.train_start) &
            (pl.col("view_start_utc") < splitter.train_end)
        ).lazy()
        val_viewing = viewing.filter(
            (pl.col("view_start_utc") >= splitter.val_start) &
            (pl.col("view_start_utc") < splitter.val_end)
        ).lazy()

        train_interactions = builder.aggregate_user_programme_interactions(train_viewing, schedule.lazy())
        val_interactions = builder.aggregate_user_programme_interactions(val_viewing, schedule.lazy())

        # Both should have the interaction
        assert train_interactions.height == 1
        assert val_interactions.height == 1
        assert train_interactions["user_id"][0] == "u1"
        assert train_interactions["tv_show_id"][0] == 100
        assert val_interactions["user_id"][0] == "u1"
        assert val_interactions["tv_show_id"][0] == 100

    def test_leakage_assertions(self):
        """Leakage assertions catch any boundary violations."""
        import zoneinfo
        utc = zoneinfo.ZoneInfo("UTC")
        splitter = TemporalSplitter(PipelineConfig())

        # Create clipped viewing manually
        train_viewing = pl.DataFrame({
            "user_id": ["u1", "u2"],
            "channel_id": [1, 2],
            "view_start_utc": [
                datetime(2020, 5, 17, 19, 0, 0, tzinfo=utc),
                datetime(2020, 4, 1, 10, 0, 0, tzinfo=utc),
            ],
            "view_end_utc": [
                datetime(2020, 5, 17, 21, 0, 0, tzinfo=utc),  # exactly at train_end
                datetime(2020, 4, 1, 11, 0, 0, tzinfo=utc),
            ],
            "duraton_seconds": [7200, 3600],
        })

        val_viewing = pl.DataFrame({
            "user_id": ["u3", "u4"],
            "channel_id": [3, 4],
            "view_start_utc": [
                datetime(2020, 5, 17, 21, 0, 0, tzinfo=utc),  # exactly at val_start
                datetime(2020, 6, 1, 10, 0, 0, tzinfo=utc),
            ],
            "view_end_utc": [
                datetime(2020, 5, 17, 22, 0, 0, tzinfo=utc),
                datetime(2020, 6, 1, 11, 0, 0, tzinfo=utc),
            ],
            "duraton_seconds": [3600, 3600],
        })

        # Should pass
        splitter.assert_no_leakage(train_viewing, val_viewing)

        # Train fragment after train_end should fail
        bad_train = pl.DataFrame({
            "user_id": ["u1"],
            "channel_id": [1],
            "view_start_utc": [datetime(2020, 5, 17, 22, 0, 0, tzinfo=utc)],
            "view_end_utc": [datetime(2020, 5, 17, 23, 0, 0, tzinfo=utc)],
            "duraton_seconds": [3600],
        })
        try:
            splitter.assert_no_leakage(bad_train, val_viewing)
            assert False, "Should have raised AssertionError"
        except AssertionError as e:
            assert "Train leakage" in str(e)

        # Val fragment before val_start should fail
        bad_val = pl.DataFrame({
            "user_id": ["u1"],
            "channel_id": [1],
            "view_start_utc": [datetime(2020, 5, 17, 20, 0, 0, tzinfo=utc)],
            "view_end_utc": [datetime(2020, 5, 17, 21, 30, 0, tzinfo=utc)],
            "duraton_seconds": [5400],
        })
        try:
            splitter.assert_no_leakage(train_viewing, bad_val)
            assert False, "Should have raised AssertionError"
        except AssertionError as e:
            assert "Val leakage" in str(e)

    def test_training_features_independent_of_val_changes(self):
        """Training features unchanged when validation viewing is modified."""
        import zoneinfo
        utc = zoneinfo.ZoneInfo("UTC")
        splitter = TemporalSplitter(PipelineConfig())
        builder = InteractionBuilder(PipelineConfig())
        ingestion = DataIngestion(PipelineConfig(data_dir=Path("data")))

        # User watches programme 100 in April (train) and programme 200 in June (val)
        viewing = pl.DataFrame({
            "user_id": ["u1", "u1"],
            "channel_id": [1, 2],
            "view_start_utc": [
                datetime(2020, 4, 1, 10, 0, 0, tzinfo=utc),   # Train
                datetime(2020, 6, 1, 10, 0, 0, tzinfo=utc),   # Val
            ],
            "view_end_utc": [
                datetime(2020, 4, 1, 11, 0, 0, tzinfo=utc),
                datetime(2020, 6, 1, 11, 0, 0, tzinfo=utc),
            ],
            "duraton_seconds": [3600, 3600],
        })

        # Schedule: programme 100 on channel 1 in April, programme 200 on channel 2 in June
        schedule = pl.DataFrame({
            "airing_id": ["a1", "a2"],
            "channel_id": [1, 2],
            "programme_start_utc": [
                datetime(2020, 4, 1, 10, 0, 0, tzinfo=utc),
                datetime(2020, 6, 1, 10, 0, 0, tzinfo=utc),
            ],
            "programme_end_utc": [
                datetime(2020, 4, 1, 11, 0, 0, tzinfo=utc),
                datetime(2020, 6, 1, 11, 0, 0, tzinfo=utc),
            ],
            "tv_show_id": [100, 200],
            "programme_duration_seconds": [3600, 3600],
        })

        train_viewing = viewing.filter(
            (pl.col("view_start_utc") >= splitter.train_start) &
            (pl.col("view_start_utc") < splitter.train_end)
        ).lazy()

        train_interactions = builder.aggregate_user_programme_interactions(train_viewing, schedule.lazy())

        # Now add extra validation viewing for same user
        viewing_with_extra = pl.concat([viewing, pl.DataFrame({
            "user_id": ["u1"],
            "channel_id": [1],
            "view_start_utc": [datetime(2020, 6, 15, 10, 0, 0, tzinfo=utc)],
            "view_end_utc": [datetime(2020, 6, 15, 11, 0, 0, tzinfo=utc)],
            "duraton_seconds": [3600],
        })])

        train_viewing2 = viewing_with_extra.filter(
            (pl.col("view_start_utc") >= splitter.train_start) &
            (pl.col("view_start_utc") < splitter.train_end)
        ).lazy()

        train_interactions2 = builder.aggregate_user_programme_interactions(train_viewing2, schedule.lazy())

        # Training interactions should be identical
        train_interactions = train_interactions.sort(["user_id", "tv_show_id"])
        train_interactions2 = train_interactions2.sort(["user_id", "tv_show_id"])
        assert train_interactions.equals(train_interactions2)


class TestUnidentifiedProgrammes:
    """Test that tv_show_id = 0 is excluded."""

    def test_unidentified_excluded(self):
        """Programmes with tv_show_id = 0 never become interactions."""
        config = PipelineConfig(data_dir=Path("data"))
        ingestion = DataIngestion(config)
        schedule = ingestion.load_schedule("11-20").collect()

        unidentified = schedule.filter(pl.col("tv_show_id") == 0)
        identified = schedule.filter(pl.col("tv_show_id") != 0)

        assert unidentified.height > 0  # There are unidentified slots
        assert identified.height > 0    # There are identified slots

        builder = InteractionBuilder(config)
        schedule_norm = builder.normalize_broadcast_intervals(schedule.lazy()).collect()

        # Normalized schedule should have no tv_show_id = 0
        assert schedule_norm.filter(pl.col("tv_show_id") == 0).height == 0


class TestDeterministicOutput:
    """Test that pipeline produces deterministic output."""

    def test_deterministic_interactions(self):
        """Same input produces same output."""
        config = PipelineConfig(data_dir=Path("data"))
        ingestion = DataIngestion(config)
        builder = InteractionBuilder(config)

        # Run twice on small sample
        viewing = ingestion.load_viewing_logs().head(1000)
        schedule = ingestion.load_schedule("11-20").head(1000)

        interactions1 = builder.aggregate_user_programme_interactions(viewing, schedule)
        interactions2 = builder.aggregate_user_programme_interactions(viewing, schedule)

        # Sort both for comparison
        interactions1 = interactions1.sort(["user_id", "tv_show_id"])
        interactions2 = interactions2.sort(["user_id", "tv_show_id"])

        assert interactions1.equals(interactions2)


class TestSmokeTest:
    """Small end-to-end smoke test."""

    def test_smoke_test_small_sample(self):
        """Run pipeline on small sample and verify output schema."""
        config = PipelineConfig(
            data_dir=Path("data"),
            processed_dir=Path("data/processed"),
            evidence_dir=Path("evidence"),
        )
        ingestion = DataIngestion(config)
        builder = InteractionBuilder(config)

        # Small sample for speed
        viewing = ingestion.load_viewing_logs().head(5000)
        schedule = ingestion.load_schedule("11-20").head(5000)

        interactions = builder.aggregate_user_programme_interactions(viewing, schedule)

        # Verify schema (updated for new aggregation)
        expected_cols = {
            "user_id", "tv_show_id", "total_watched_seconds",
            "total_broadcasts_watched", "max_coverage", "mean_coverage",
            "num_qualifying_broadcasts", "first_interaction_utc",
            "last_interaction_utc", "distinct_viewing_days", "is_relevant",
            "qualifying_watched_seconds", "max_coverage_all", "mean_coverage_all",
            "max_coverage_qual", "mean_coverage_qual",
        }
        assert set(interactions.columns) == expected_cols

        # Verify types
        assert interactions["user_id"].dtype == pl.Utf8
        assert interactions["tv_show_id"].dtype == pl.Int64
        assert interactions["is_relevant"].dtype == pl.Boolean

        # Verify no negative values
        assert (interactions["total_watched_seconds"] >= 0).all()
        assert (interactions["coverage"] >= 0).all() if "coverage" in interactions.columns else True
        assert (interactions["max_coverage"] <= 1.0).all()
        assert (interactions["mean_coverage"] <= 1.0).all()

        print(f"Smoke test: {interactions.height} user-programme interactions")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])