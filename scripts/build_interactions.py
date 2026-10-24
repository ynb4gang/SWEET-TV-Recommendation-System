#!/usr/bin/env python3
"""
Script to build user-programme interactions from raw data.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.interaction_pipeline import PipelineConfig, run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    config = PipelineConfig(
        data_dir=Path("data"),
        processed_dir=Path("data/processed"),
        evidence_dir=Path("evidence"),
    )
    evidence = run_pipeline(config)
    print(f"Pipeline completed successfully: {evidence}")