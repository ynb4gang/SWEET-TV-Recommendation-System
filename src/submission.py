"""Strict prediction and independent standard-library CSV verification."""
import csv
import hashlib
import re
from pathlib import Path


def template_users(path="data/submission.csv"):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["user_id", "tv_show_id"]:
            raise ValueError("Unexpected template header")
        users = [r["user_id"] for r in reader]
    if len(users) != len(set(users)) or any(not re.fullmatch(r"[0-9]+", u) for u in users):
        raise ValueError("Invalid or duplicate template users")
    return users


def validate_predictions(predictions, users, eligible, k=5):
    if len(users) != len(set(users)) or any(not isinstance(u, str) for u in users):
        raise ValueError("User IDs must be unique exact strings")
    if set(predictions) != set(users):
        raise ValueError("Missing or extra users")
    for user in users:
        items = predictions[user]
        if len(items) != k or len(set(items)) != k:
            raise ValueError(f"Invalid recommendation length/duplicates: {user}")
        if any(not isinstance(p, int) or isinstance(p, bool) or p <= 0 or p not in eligible for p in items):
            raise ValueError(f"Unknown, zero, or ineligible programme: {user}")
    return True


def write_submission(path, predictions, eligible, template="data/submission.csv"):
    users = template_users(template)
    validate_predictions(predictions, users, eligible)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite a previously secured submission with different predictions.
    if path.exists():
        with path.open(newline="") as f:
            old = list(csv.DictReader(f))
        expected = [{"user_id": u, "tv_show_id": " ".join(map(str, predictions[u]))} for u in users]
        if old != expected:
            raise FileExistsError(f"Refusing to overwrite {path}; archive it explicitly first")
    else:
        with path.open("x", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["user_id", "tv_show_id"])
            for user in users:
                writer.writerow([user, " ".join(map(str, predictions[user]))])
    return validate_csv(path, eligible, template)


def validate_csv(path, eligible, template="data/submission.csv"):
    """Re-read bytes through csv, not the model or DataFrame serialization code."""
    users = template_users(template)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["user_id", "tv_show_id"]:
            raise ValueError("Incorrect CSV header")
        rows = list(reader)
    if [r["user_id"] for r in rows] != users:
        raise ValueError("CSV users/order differ from template")
    predictions = {}
    for row in rows:
        if not re.fullmatch(r"[1-9][0-9]*( [1-9][0-9]*){4}", row["tv_show_id"]):
            raise ValueError("Malformed predictions")
        predictions[row["user_id"]] = list(map(int, row["tv_show_id"].split(" ")))
    validate_predictions(predictions, users, eligible)
    return {"valid": True, "rows": len(rows), "exact_user_order": True,
            "eligible_programmes": len(eligible),
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
