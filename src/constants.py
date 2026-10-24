"""
Shared constants and config for the SWEET.TV pipeline.
"""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import zoneinfo

# Timezone for source data (Ukraine/Kyiv - EET/EEST)
KYIV_TZ = "Europe/Kyiv"
_KYIV = zoneinfo.ZoneInfo(KYIV_TZ)
_UTC = zoneinfo.ZoneInfo("UTC")

# Temporal split boundaries defined in Europe/Kyiv local time, then converted to UTC
# 2020-03-09 00:00 Kyiv (EET, UTC+2) = 2020-03-08 22:00 UTC
_TRAIN_START_KYIV = datetime(2020, 3, 9, 0, 0, 0, tzinfo=_KYIV)
TRAIN_START = _TRAIN_START_KYIV.astimezone(_UTC)

# 2020-05-18 00:00 Kyiv (EEST, UTC+3) = 2020-05-17 21:00 UTC
_TRAIN_END_KYIV = datetime(2020, 5, 18, 0, 0, 0, tzinfo=_KYIV)
TRAIN_END = _TRAIN_END_KYIV.astimezone(_UTC)
VAL_START = TRAIN_END  # Same boundary

# 2020-07-27 00:00 Kyiv (EEST, UTC+3) = 2020-07-26 21:00 UTC
_VAL_END_KYIV = datetime(2020, 7, 27, 0, 0, 0, tzinfo=_KYIV)
VAL_END = _VAL_END_KYIV.astimezone(_UTC)

# 2020-07-27 00:00 Kyiv (EEST, UTC+3) = 2020-07-26 21:00 UTC
_TEST_START_KYIV = datetime(2020, 7, 27, 0, 0, 0, tzinfo=_KYIV)
TEST_START = _TEST_START_KYIV.astimezone(_UTC)

# 2020-10-20 00:00 Kyiv (EEST, UTC+3) = 2020-10-19 21:00 UTC
_TEST_END_KYIV = datetime(2020, 10, 20, 0, 0, 0, tzinfo=_KYIV)
TEST_END = _TEST_END_KYIV.astimezone(_UTC)

# Minimum broadcast completion for positive relevance
COVERAGE_THRESHOLD = 0.80


@dataclass
class PipelineConfig:
    """Configuration for the interaction pipeline."""
    data_dir: Path = Path("data")
    processed_dir: Path = Path("data/processed")
    evidence_dir: Path = Path("evidence")
    coverage_threshold: float = COVERAGE_THRESHOLD
    use_cache: bool = True