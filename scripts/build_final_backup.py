"""Secure full-history Recency30 backup before further experiments."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.optimization import window_interactions, save_json, msk_now
from src.constants import TRAIN_START, TEST_START
from src.baselines import RecencyWeightedRecommender, get_test_eligible_programmes
from src.submission import template_users, write_submission

if __name__ == "__main__":
    history = window_interactions("full_history", TRAIN_START, TEST_START)
    eligible = get_test_eligible_programmes()
    model = RecencyWeightedRecommender(half_life_days=30, cutoff=TEST_START).fit(history, eligible)
    users = template_users()
    recs = model.recommend_batch(users)
    check = write_submission("outputs/submission_recency30_backup.csv", recs, eligible)
    save_json("evidence/phase05_first_submission.json", {
        "created_msk": msk_now(), "model": "Recency30", "cutoff_utc": TEST_START,
        "history_pairs": history.height, "half_life_days": 30,
        "recency": "aggregate last-interaction proxy", "csv": check,
        "kaggle": "Not submitted; manual upload requested"})
    print(check)
