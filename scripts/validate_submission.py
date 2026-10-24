"""Independent CSV validation entry point; no Kaggle access."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baselines import get_test_eligible_programmes
from src.submission import validate_csv
import json

if __name__ == "__main__":
    print(json.dumps({p: validate_csv(p, get_test_eligible_programmes()) for p in sys.argv[1:]}, indent=2))
