"""Bounded objective-alignment experiment; preserves all older artifacts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gc
import json
import time
from hashlib import sha256

import lightgbm as lgb
import numpy as np
import polars as pl

from scripts.phase06_optimize import snapshot, interactions, date
from src.constants import TRAIN_START, TRAIN_END, VAL_END, TEST_START, TEST_END, PipelineConfig
from src.contextual_reranking import exposure_features, context_features, ranked_predictions
from src.interaction_pipeline import DataIngestion
from src.optimization import historical_schedule, historical_viewing, raw_users, EARLY_CUTOFF, save_json, msk_now, rrf
from src.objective_alignment import (merged_airing_segments, airing_window, frequency_counts, top_five_target,
    target_sizes, dual_metrics, strength_features, retrieval, anchor_scores, blend_scores, ANCHORS)
from src.reranking import candidate_features, TemporalRanker
from src.submission import template_users, write_submission, validate_csv, validate_predictions

EVIDENCE = "evidence/phase08_target_alignment.json"


def pools_from_frame(frame):
    return {u[0]: f["tv_show_id"].to_list() for u, f in frame.partition_by("user_id", as_dict=True).items()}


def verify_counts(counts, old):
    left = counts.select("user_id", "tv_show_id", pl.col("num_qualifying_broadcasts").alias("new"))
    right = old.select("user_id", "tv_show_id", pl.col("num_qualifying_broadcasts").alias("old"))
    checked = left.join(right, on=["user_id", "tv_show_id"], how="full", coalesce=True).fill_null(0)
    return {"compared_pairs": checked.height, "different_qualifying_counts": checked.filter(pl.col("new") != pl.col("old")).height,
            "qualifying_airings": int(counts["num_qualifying_broadcasts"].sum())}


def evaluate(snap, predictions, populations, pool):
    return {name: dual_metrics(predictions, snap["all_target"], snap["top_target"], users, pool)
            for name, users in populations.items()}


def augmented(snap, segments, views, schedule):
    name, cutoff = snap["name"], snap["cutoff"]
    path = Path(f"data/processed/phase08_{name}_features.parquet")
    info = path.with_suffix(".json")
    if path.exists() and info.exists():
        meta = json.loads(info.read_text())
        snap.update(new_frame=pl.read_parquet(path), features=meta["features"], retrieval_checks=meta["retrieval_checks"])
        snap["pool"] = pools_from_frame(snap["new_frame"])
        return
    from datetime import timedelta
    t = time.perf_counter()
    past = schedule.filter((pl.col("programme_start_utc") < cutoff) & (pl.col("programme_end_utc") > TRAIN_START)).unique(
        subset=["channel_id", "programme_start_utc", "tv_show_id"])
    target_schedule = snap["schedule"]
    history_airings = airing_window(segments, TRAIN_START, cutoff)
    strength = strength_features(history_airings, cutoff)
    pool, scores, checks = retrieval(history_airings, strength, views, past, target_schedule, cutoff,
                                     snap["users"], snap["eligible"], snap["ranks"])
    ranks = {k: snap["ranks"][k] for k in ("global", "frequency", "recency", "ppr", "cbf")}
    ranks["pool"] = pool
    frame, baseline = candidate_features(snap["history"], ranks, snap["users"], snap["eligible"], cutoff, target_schedule, depth=220)
    frame = frame.join(scores.select("user_id", "tv_show_id"), on=["user_id", "tv_show_id"], how="inner")
    recent = {d: interactions(max(TRAIN_START, cutoff-timedelta(days=d)), cutoff) for d in (7, 14, 28)}
    frame, af, _ = exposure_features(frame, snap["history"], recent, views, schedule, target_schedule, TRAIN_START, cutoff, snap["end"])
    frame, bf = context_features(frame, recent, views, schedule, target_schedule, TRAIN_START, cutoff)
    frame = frame.join(strength, on=["user_id", "tv_show_id"], how="left").join(scores, on=["user_id", "tv_show_id"], how="left")
    frame = frame.with_columns(pl.col("observed_days_ago").fill_null(10000)).fill_null(0).sort(["user_id", "tv_show_id"])
    sf = [c for c in strength.columns if c not in ("user_id", "tv_show_id")]
    frame = frame.with_columns([pl.col(c).cast(pl.Float64).log1p() for c in sf if c != "observed_days_ago"])
    rf = [c for c in scores.columns if c.startswith("score_")]
    frame = frame.with_columns([(pl.col(c).log1p()/pl.col(c).log1p().max().over("user_id").clip(1e-12)).alias("normalized_"+c) for c in rf])
    features = baseline + af + bf + sf + rf + ["normalized_"+c for c in rf]
    assert len(features) == len(set(features))
    frame = frame.with_columns([pl.col(c).cast(pl.Float32) for c in features])
    assert np.isfinite(frame.select(features).to_numpy()).all()
    assert not frame.select("user_id", "tv_show_id").is_duplicated().any()
    frame.write_parquet(path)
    save_json(info, {"features": features, "retrieval_checks": checks, "cutoff": cutoff, "target_end": snap["end"]})
    snap.update(new_frame=frame, features=features, pool=pool, retrieval_checks=checks)
    print(f"new candidates {name}: {frame.height} rows, {time.perf_counter()-t:.1f}s", flush=True)


def fit(snapshots):
    features = snapshots[0]["features"]
    frames, targets = [], []
    for snap in snapshots:
        assert snap["features"] == features
        prefix = snap["name"]+":"
        frames.append(snap["new_frame"].select(["user_id", "tv_show_id"]+features).with_columns(
            (pl.lit(prefix)+pl.col("user_id")).alias("user_id")))
        targets.append(snap["top_target"].select("user_id", "tv_show_id", "is_relevant").with_columns(
            (pl.lit(prefix)+pl.col("user_id")).alias("user_id")))
    return TemporalRanker().fit(pl.concat(frames), features, pl.concat(targets))


def main():
    t = time.perf_counter()
    protected = {str(p): sha256(p.read_bytes()).hexdigest() for root in ("outputs", "evidence")
                 for p in Path(root).glob("*") if p.is_file() and not p.name.startswith("phase08")
                 and p.name not in ("submission_objective_v5.csv", "submission_anchor_v5.csv")}
    evidence = {"created_msk": msk_now(), "target_hypothesis": "Top-5 most frequently >=80%-completed distinct airings; official Kaggle definition NOT confirmed",
        "definitions": {"all_positive": "Every programme with at least one >=80% completed airing in the window; unchanged old metric",
            "top5_frequency": "Distinct qualifying airing count descending, numeric programme ID ascending, retain <=5 per user; empty users contribute zero"},
        "methodology": {"cohorts": "Fixed 4876 raw training-viewing users and all 1260 exact template users, independently reported",
            "segments": "Union overlapping viewing fragments per user-airing, clip observed seconds to snapshot, retain complete airing duration denominator",
            "anchor_grid": {"weights_order": ["exposure", "60-day-decayed item-neighbor propagation", "PPR reciprocal rank", "channel/time", "schedule recurrence"],
                            "weights": ANCHORS, "smoothing": [10, 80]},
            "candidate_pool": "220, round-robin union of frequency, exposure10/80, recency, PPR, item neighbors, context, popularity; no target-guided insertion",
            "neighbors": "Meaningful >=50% coverage AND >=60 seconds; binary cosine co-viewing, no self-links, top80 neighbors; all historical users in graph",
            "exposure": "qualifying_view_weight=1 for each distinct >=80% airing; sum n80(user,item,channel)*future_airings(item,channel)/(past_airings(item,channel)+smoothing)",
            "training": "One fixed contextual LambdaRank configuration, trained on the same Top-5 target as evaluation; B is a score blend, not another architecture",
            "selection": "Mean submission-cohort Top5 MAP across two disjoint chronological backtests; prefer configurations improving both versus current PPR; original long split diagnostic only",
            "caveats": ["Long main target overlaps both short backtests", "Kaggle relevance definition and future viewing labels unavailable",
                        "Only the saved early-trained LightGBM can be fairly rescored before any training; production phase-six model contains validation labels and is excluded"]},
        "target_verification": {}, "before_training": {}, "results": {}, "candidate_recall": {}}
    schedule = historical_schedule().collect()
    views = historical_viewing().collect()
    submission = template_users()
    population = raw_users(TRAIN_START, TRAIN_END)
    populations = {"all": population, "submission": submission}
    segments = merged_airing_segments(views, schedule)
    early = snapshot("early", EARLY_CUTOFF, TRAIN_END, schedule, views, submission)
    short = snapshot("short", TRAIN_END, date(6, 22), schedule, views, submission)
    late = snapshot("late", date(6, 22), VAL_END, schedule, views, submission)
    main_snap = snapshot("main", TRAIN_END, VAL_END, schedule, views, submission)
    snaps = [early, short, late, main_snap]
    validations = [short, late, main_snap]
    evidence["windows"] = {s["name"]: {"history_start": TRAIN_START, "cutoff": s["cutoff"], "target_end_exclusive": s["end"],
        "ranker_training_snapshots": ["early", "short"] if s["name"] == "late" else ["early"]} for s in validations}
    for snap in snaps:
        target_airings = airing_window(segments, snap["cutoff"], snap["end"])
        counts = frequency_counts(target_airings)
        evidence["target_verification"][snap["name"]] = verify_counts(counts, snap["targets"])
        snap["all_target"], snap["top_target"] = counts, top_five_target(counts)
        snap["schedule"] = schedule.filter((pl.col("programme_start_utc") < snap["end"]) & (pl.col("programme_end_utc") > snap["cutoff"]) & (pl.col("tv_show_id") > 0)).unique(
            subset=["channel_id", "programme_start_utc", "tv_show_id"])
    del target_airings
    # Mandatory diagnostic happens before co-viewing fitting or supervised training.
    old_model = lgb.Booster(model_file="outputs/phase05_ranker.txt")
    for snap in validations:
        name, users = snap["name"], snap["users"]
        old_pool = pools_from_frame(snap["frame"])
        recs = {k: {u: snap["ranks"][k][u][:5] for u in users} for k in ("recency", "ppr")}
        recs["hybrid"] = rrf([snap["ranks"][k] for k in ("recency", "ppr", "global")], [.4, .4, .2], users, snap["eligible"])
        margins = old_model.predict(snap["frame"].select(snap["groups"]["baseline"]).to_numpy().astype(np.float32))
        recs["existing_lightgbm"] = ranked_predictions(snap["frame"], margins, users, snap["eligible"])
        evidence["before_training"][name] = {"models": {k: evaluate(snap, r, populations, old_pool) for k, r in recs.items()},
            "target_sizes": {cohort: {"all_positive": target_sizes(snap["all_target"], ids), "top5_frequency": target_sizes(snap["top_target"], ids)} for cohort, ids in populations.items()}}
        evidence["candidate_recall"][name] = {"before": evaluate(snap, recs["existing_lightgbm"], populations, old_pool)}
        evidence["results"][name] = {k: evaluate(snap, r, populations, old_pool) for k, r in recs.items()}
        r = evidence["results"][name]["existing_lightgbm"]["submission"]
        print(f"before training {name}: old LightGBM all-positive={r['all_positive']['map5']:.6f}, Top5={r['top5_frequency']['map5']:.6f}", flush=True)
    evidence["before_training_completed_msk"] = msk_now()
    save_json(EVIDENCE, evidence)
    for snap in snaps:
        del snap["frame"]
        snap.pop("profiles", None)
    gc.collect()

    for snap in snaps:
        augmented(snap, segments, views, schedule)
    anchor_options = {}
    for smoothing in (10, 80):
        for variant in ANCHORS:
            key = f"anchor_{smoothing}_{variant}"
            anchor_options[key] = (smoothing, variant)
            for snap in validations:
                values = anchor_scores(snap["new_frame"], smoothing, variant)
                recs = ranked_predictions(snap["new_frame"], values, snap["users"], snap["eligible"])
                evidence["results"][snap["name"]][key] = evaluate(snap, recs, populations, snap["pool"])
    def metric(window, model):
        return evidence["results"][window][model]["submission"]["top5_frequency"]["map5"]
    def select(names, reference="ppr"):
        consistent = [n for n in names if all(metric(w, n) >= metric(w, reference) for w in ("short", "late"))]
        candidates = consistent or list(names)
        return max(candidates, key=lambda n: (np.mean([metric(w, n) for w in ("short", "late")]), n)), consistent
    anchor_key, consistent_anchors = select(list(anchor_options))
    anchor_config = anchor_options[anchor_key]
    evidence["selected_anchor"] = {"name": anchor_key, "smoothing": anchor_config[0], "variant": anchor_config[1], "consistent_options": consistent_anchors}
    for snap in validations:
        evidence["candidate_recall"][snap["name"]]["after"] = evidence["results"][snap["name"]][anchor_key]
    # Fit only model A; B compares two small weights under two stable transformations.
    for snap in validations:
        training = [early, short] if snap["name"] == "late" else [early]
        assert max(s["end"] for s in training) <= snap["cutoff"]
        model = fit(training)
        values = model.model.predict(snap["new_frame"].select(model.features).to_numpy().astype(np.float32))
        recs = ranked_predictions(snap["new_frame"], values, snap["users"], snap["eligible"])
        evidence["results"][snap["name"]]["A_lightgbm"] = evaluate(snap, recs, populations, snap["pool"])
        av = anchor_scores(snap["new_frame"], *anchor_config)
        for transform in ("rank", "exponential"):
            for weight in (.2, .5):
                key = f"B_{transform}_{weight}"
                blended = blend_scores(snap["new_frame"], values, av, weight, transform)
                predictions = ranked_predictions(snap["new_frame"], blended, snap["users"], snap["eligible"])
                evidence["results"][snap["name"]][key] = evaluate(snap, predictions, populations, snap["pool"])
        for key in ("recency", "ppr", anchor_key, "A_lightgbm", "B_rank_0.2", "B_exponential_0.2"):
            rs = evidence["results"][snap["name"]][key]["submission"]
            print(f"{snap['name']} {key}: Top5={rs['top5_frequency']['map5']:.6f}, all-positive={rs['all_positive']['map5']:.6f}", flush=True)
        save_json(EVIDENCE, evidence)
    options = [anchor_key, "A_lightgbm"] + [f"B_{tr}_{w}" for tr in ("rank", "exponential") for w in (.2, .5)]
    selected, consistent = select(options)
    evidence["selection"] = {"objective": selected, "anchor": anchor_key, "consistent_options": consistent,
        "two_nonoverlapping_windows_only": True, "main_diagnostic_not_used_for_selection": True}
    del main_snap["new_frame"]
    gc.collect()
    # Selection frozen: legitimate full-history refit on three disjoint label windows.
    production = fit([early, short, late])
    production.model.booster_.save_model("outputs/phase08_objective_v5.txt")
    future = DataIngestion(PipelineConfig()).load_schedule("31-42").collect()
    final = snapshot("final", TEST_START, TEST_END, schedule, views, submission, future=future)
    final["schedule"] = future.filter((pl.col("programme_start_utc") < TEST_END) & (pl.col("programme_end_utc") > TEST_START) & (pl.col("tv_show_id") > 0)).unique(
        subset=["channel_id", "programme_start_utc", "tv_show_id"])
    augmented(final, segments, views, schedule)
    av = anchor_scores(final["new_frame"], *anchor_config)
    values = production.model.predict(final["new_frame"].select(production.features).to_numpy().astype(np.float32))
    def predictions():
        objective = av if selected == anchor_key else values
        if selected.startswith("B_"):
            _, transform, weight = selected.split("_")
            objective = blend_scores(final["new_frame"], values, av, float(weight), transform)
        return ranked_predictions(final["new_frame"], objective, submission, final["eligible"])
    objective = predictions()
    anchor = ranked_predictions(final["new_frame"], av, submission, final["eligible"])
    assert objective == predictions()
    assert anchor == ranked_predictions(final["new_frame"], anchor_scores(final["new_frame"], *anchor_config), submission, final["eligible"])
    checks = {}
    for path, recs in (("outputs/submission_objective_v5.csv", objective), ("outputs/submission_anchor_v5.csv", anchor)):
        validate_predictions(recs, submission, final["eligible"])
        write_submission(path, recs, final["eligible"])
        checks[path] = validate_csv(path, final["eligible"])
    for path, digest in protected.items():
        assert sha256(Path(path).read_bytes()).hexdigest() == digest, path
    evidence.update({"completed_msk": msk_now(), "csv_checks": checks, "deterministic_predictions": True,
        "protected_artifacts_unchanged": True, "protected_sha256": protected,
        "code_sha256": {p: sha256(Path(p).read_bytes()).hexdigest() for p in ("src/objective_alignment.py", "scripts/phase08_align.py")},
        "production": {"training_rows": production.training_rows, "snapshot_queries": production.training_users,
            "label_windows": [[s["cutoff"], s["end"]] for s in (early, short, late)], "features": production.features,
            "prediction_cutoff": TEST_START, "prediction_end": TEST_END, "refit_evaluated_on_training_labels": False},
        "runtime_seconds": time.perf_counter()-t, "submitted_to_kaggle": False})
    save_json(EVIDENCE, evidence)
    print(f"Selected {selected}; anchor {anchor_key}. CSVs validated, prior artifacts unchanged.", flush=True)


if __name__ == "__main__":
    main()
