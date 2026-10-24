#!/usr/bin/env python3
"""
Script to run CBF benchmark evaluation.
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl

from src.content_based import run_cbf_benchmark
from src.constants import PipelineConfig
from src.interaction_pipeline import DataIngestion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    config = PipelineConfig()

    # Load pre-computed interactions
    train_path = config.processed_dir / "train_interactions.parquet"
    val_path = config.processed_dir / "val_interactions.parquet"

    if not train_path.exists() or not val_path.exists():
        logger.error("Processed interaction files not found. Run build_interactions.py first.")
        sys.exit(1)

    logger.info("Loading training interactions...")
    train_interactions = pl.read_parquet(train_path)

    logger.info("Loading validation interactions...")
    val_interactions = pl.read_parquet(val_path)

    logger.info(f"Train interactions: {train_interactions.height}")
    logger.info(f"Validation interactions: {val_interactions.height}")

    # Load schedule
    logger.info("Loading schedule data...")
    ingestion = DataIngestion(config)
    schedule_lf = ingestion.load_all_schedules()

    # Run benchmark
    logger.info("Starting CBF benchmark...")
    results = run_cbf_benchmark(
        train_interactions,
        val_interactions,
        schedule_lf,
        "data/submission.csv"
    )

    # Save results
    evidence_dir = config.evidence_dir
    evidence_dir.mkdir(parents=True, exist_ok=True)
    output_path = evidence_dir / "phase04_cbf.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 100)
    print("CBF BENCHMARK RESULTS")
    print("=" * 100)
    print(f"{'Model':<35} {'MAP@5 All':>10} {'MAP@5 Sub':>10} {'HitRate@5':>10} {'Coverage':>10} {'Runtime(s)':>10} {'Eligible':>10}")
    print("-" * 100)

    for model, stats in results.items():
        print(f"{model:<35} {stats['map_at_k_all']:>10.6f} "
              f"{stats['map_at_k_submission']:>10.6f} "
              f"{stats['hit_rate_at_k_all']:>10.6f} "
              f"{stats['candidate_coverage']:>10.4f} "
              f"{stats['runtime_seconds']:>10.2f} "
              f"{stats['eligible_programmes_count']:>10}")

    print("=" * 100)

    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]["map_at_k_all"])
    print(f"\nBest CBF model (MAP@5 all): {best_model[0]} with MAP@5 = {best_model[1]['map_at_k_all']:.6f}")

    # Report population stats
    first = list(results.values())[0]
    print(f"\nPopulation A (all known users): {first['num_evaluation_users_all']}")
    print(f"  - With relevant items: {first['num_users_with_relevant_all']}")
    print(f"  - Zero relevant items: {first['num_users_zero_relevant_all']}")
    print(f"Population B (submission users): {first['num_evaluation_users_submission']}")
    print(f"  - With relevant items: {first['num_users_with_relevant_submission']}")
    print(f"  - Zero relevant items: {first['num_users_zero_relevant_submission']}")
    print(f"Eligible programmes (schedule-derived): {first['eligible_programmes_count']}")


if __name__ == "__main__":
    main()