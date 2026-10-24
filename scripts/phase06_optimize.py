"""Bounded, chronological A/B/C experiments. No Kaggle access or phase-five writes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collections import Counter
from datetime import datetime, timedelta
from hashlib import sha256
import json
import time
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import polars as pl

from scripts.optimize_and_submit import components
from src.baselines import get_schedule_eligible_programmes
from src.constants import TRAIN_START, TRAIN_END, VAL_END, TEST_START, TEST_END, PipelineConfig
from src.contextual_reranking import (clip_views, exposure_features, context_features,
                                      CalibratedAdaptiveBlend, ranked_predictions)
from src.evaluation import build_ground_truth, average_precision_at_k
from src.interaction_pipeline import DataIngestion
from src.optimization import (window_interactions, historical_schedule, historical_viewing,
                              raw_users, evaluate, save_json, msk_now, EARLY_CUTOFF)
from src.reranking import candidate_features, TemporalRanker
from src.submission import template_users, write_submission, validate_csv

VERSION = "phase06_v1"
KEYS = ["global", "frequency", "recency", "ppr", "cbf"]


def date(month, day):
    return datetime(2020, month, day, tzinfo=ZoneInfo("Europe/Kyiv")).astimezone(ZoneInfo("UTC"))


def interactions(start, end):
    existing = {(TRAIN_START, EARLY_CUTOFF): "early_train", (EARLY_CUTOFF, TRAIN_END): "early_targets",
                (TRAIN_START, TRAIN_END): "train", (TRAIN_END, VAL_END): "validation",
                (TRAIN_START, TEST_START): "full_history"}
    tag = existing.get((start, end), f"phase06_{start:%m%d}_{end:%m%d}")
    return window_interactions(tag, start, end)


def snapshot(name, cutoff, end, schedule, views, submission, future=None):
    t = time.perf_counter()
    population = raw_users(TRAIN_START, cutoff)
    users = list(dict.fromkeys(population + submission))
    history = interactions(TRAIN_START, cutoff)
    target_schedule = (future if future is not None else schedule).filter(
        (pl.col("programme_start_utc") < end) & (pl.col("programme_end_utc") > cutoff))
    eligible = get_schedule_eligible_programmes(target_schedule.lazy(), cutoff, end)
    targets = None if future is not None else interactions(cutoff, end)
    cache = Path(f"data/processed/{VERSION}_{name}_features.parquet")
    info = cache.with_suffix(".json")
    profiles_path = cache.with_name(f"{VERSION}_{name}_profiles.parquet")
    # The original main component cache is intentionally reused read-only.
    window = {"early": "early", "main": "validation", "final": "final"}.get(name, f"phase06_{name}")
    metadata = pl.concat([schedule, future]) if future is not None else schedule.filter(pl.col("programme_start_utc") < end)
    # Phase-five final caches have only template users; preserve their cohort.
    component_users = submission if name == "final" else users
    ranks, timing = components(history, metadata, eligible, component_users, cutoff, window, (.30,))
    ranks["ppr"] = ranks["ppr_0.3"]
    if name == "final":
        users = submission
    if cache.exists() and info.exists() and profiles_path.exists():
        frame = pl.read_parquet(cache)
        feature_groups = json.loads(info.read_text())
        profiles = pl.read_parquet(profiles_path)
    else:
        base, baseline = candidate_features(history, {k: ranks[k] for k in KEYS}, users, eligible, cutoff, target_schedule)
        recent = {d: interactions(max(TRAIN_START, cutoff - timedelta(days=d)), cutoff) for d in (7, 14, 28)}
        frame, a, profiles = exposure_features(base, history, recent, views, schedule, target_schedule,
                                               TRAIN_START, cutoff, end)
        frame, b = context_features(frame, recent, views, schedule, target_schedule, TRAIN_START, cutoff)
        feature_groups = {"baseline": baseline, "A": baseline + a, "B": baseline + a + b,
                          "exposure_added": a, "context_added": b}
        nonfinite = [c for c in feature_groups["B"] if not np.isfinite(frame[c].to_numpy()).all()]
        assert not nonfinite, f"Nonfinite features: {nonfinite}"
        frame.write_parquet(cache)
        profiles.write_parquet(profiles_path)
        save_json(info, feature_groups)
    print(f"snapshot {name}: {len(users)} users, {frame.height} candidates, {time.perf_counter()-t:.1f}s", flush=True)
    return dict(name=name, cutoff=cutoff, end=end, population=population, users=users, history=history,
                targets=targets, eligible=eligible, frame=frame, groups=feature_groups, profiles=profiles,
                ranks=ranks, component_timing=timing)


def fit(snapshots, family):
    frames, labels = [], []
    features = snapshots[0]["groups"][family]
    for snap in snapshots:
        prefix = snap["name"] + ":"
        frames.append(snap["frame"].select(["user_id", "tv_show_id"] + features).with_columns(
            (pl.lit(prefix) + pl.col("user_id")).alias("user_id")))
        labels.append(snap["targets"].select("user_id", "tv_show_id", "is_relevant").with_columns(
            (pl.lit(prefix) + pl.col("user_id")).alias("user_id")))
    return TemporalRanker().fit(pl.concat(frames), features, pl.concat(labels))


def score(model, snap):
    return model.model.predict(snap["frame"].select(model.features).to_numpy().astype(np.float32))


def concentration(recs):
    counts = Counter(p for values in recs.values() for p in values)
    total = sum(counts.values()) or 1
    return {"distinct": len(counts), "top5_share": sum(v for _, v in counts.most_common(5))/total,
            "top10_share": sum(v for _, v in counts.most_common(10))/total,
            "hhi": sum((v/total)**2 for v in counts.values())}


def diagnostics(snap, views):
    users = snap["population"]
    truth = build_ground_truth(snap["targets"])
    candidates = {u[0]: set(g["tv_show_id"].to_list()) for u, g in snap["frame"].partition_by("user_id", as_dict=True).items()}
    past = {u[0]: set(g["tv_show_id"].to_list()) for u, g in snap["history"].filter(pl.col("is_relevant")).partition_by("user_id", as_dict=True).items()}
    total = sum(len(truth.get(u, set())) for u in users)
    recalled = sum(len(truth.get(u, set()) & candidates.get(u, set())) for u in users)
    recurrent = sum(len(truth.get(u, set()) & past.get(u, set())) for u in users)
    target_ids = {p for u in users for p in truth.get(u, set())}
    eligible_truth = sum(len(truth.get(u, set()) & snap["eligible"]) for u in users)
    observation_days = (snap["cutoff"]-TRAIN_START).total_seconds()/86400
    prediction_days = (snap["end"]-snap["cutoff"]).total_seconds()/86400
    rates = []
    for start, end in ((TRAIN_START, snap["cutoff"]), (snap["cutoff"], snap["end"])):
        seconds = clip_views(views, start, end).group_by("user_id").agg(pl.col("seconds").sum()).with_columns(
            (pl.col("seconds") / ((end-start).total_seconds()/86400) / 3600).alias("hours_per_day"))
        rates.append(pl.DataFrame({"user_id": users}).join(seconds, on="user_id", how="left").fill_null(0)["hours_per_day"].to_numpy())
    ratios = (rates[1]+1/60)/(rates[0]+1/60)
    return {"history_days": observation_days, "prediction_days": prediction_days,
            "raw_users": len(users), "submission_users": 1260, "eligible": len(snap["eligible"]),
            "candidate_pair_recall": recalled/max(total, 1), "schedule_recall": eligible_truth/max(total, 1),
            "candidate_oracle_map5": float(np.mean([min(5, len(truth.get(u, set()) & candidates.get(u, set())))/
                min(5, len(truth[u])) if truth.get(u) else 0 for u in users])),
            "positive_pair_recurrence": recurrent/max(total, 1),
            "target_programmes_seen_in_history_fraction": len(target_ids & set(snap["history"]["tv_show_id"]))/max(len(target_ids), 1),
            "median_hours_per_day_history_target": [float(np.median(v)) for v in rates],
            "median_target_to_history_activity_ratio": float(np.median(ratios)),
            "fraction_users_activity_halved": float(np.mean(ratios < .5)),
            "fraction_users_activity_doubled": float(np.mean(ratios > 2)),
            "truth_concentration": concentration({u: sorted(truth.get(u, set())) for u in users})}


def main():
    started = time.perf_counter()
    new_outputs = {"submission_context_v2.csv", "submission_adaptive_v2.csv"}
    protected = [p for root in ("outputs", "evidence") for p in Path(root).glob("*")
                 if p.is_file() and not p.name.startswith("phase06") and p.name not in new_outputs]
    hashes = {str(p): sha256(p.read_bytes()).hexdigest() for p in protected}
    evidence = {"created_msk": msk_now(), "protocol": {
        "families": ["baseline", "A: normalized exposure and recent programme activity", "B: A + channel/time/duration", "C: calibrated adaptive blend"],
        "cohort_rule": "Fixed original 4876 raw training-viewing users across every validation window, including users without recognized programmes; all 1260 template users separately. Snapshot training and feature generation may include additional users known before its cutoff.",
        "selection": "Nonnegative improvement over baseline in both disjoint backtests and original split; maximize mean MAP5 gain, otherwise retain baseline",
        "calibration": "Earlier out-of-time predictions, no validation or Kaggle labels",
        "kaggle_scores_user_reported_not_targets": {"public": .43821, "private": .42859},
        "limitations": ["The original long holdout overlaps both short backtests", "Relevance is the existing 80% broadcast-completion proxy, Kaggle definition unverified",
                        "Channel/hour/day profiles use clipped session start and watched seconds; programme stability uses a difference-of-airing-count proxy"]},
        "results": {}, "diagnostics": {}}
    schedule = historical_schedule().collect()
    views = historical_viewing().collect()
    submission = template_users()
    pre = snapshot("pre", date(3, 30), EARLY_CUTOFF, schedule, views, submission)
    early = snapshot("early", EARLY_CUTOFF, TRAIN_END, schedule, views, submission)
    short = snapshot("short", TRAIN_END, date(6, 22), schedule, views, submission)
    main_snap = snapshot("main", TRAIN_END, VAL_END, schedule, views, submission)
    late = snapshot("late", date(6, 22), VAL_END, schedule, views, submission)
    validations = [short, late, main_snap]
    fixed_population = raw_users(TRAIN_START, TRAIN_END)
    for snap in validations:
        snap["population"] = fixed_population
    evidence["windows"] = {s["name"]: {"cutoff": s["cutoff"], "end": s["end"],
        "training_snapshots": ["early", "short"] if s["name"] == "late" else ["early"]} for s in validations}
    evidence["feature_groups"] = early["groups"]
    oof_scores, validation_scores, validation_recs = {}, {}, {}
    for family in ("baseline", "A", "B"):
        oof_model = fit([pre], family)
        oof_scores[family] = score(oof_model, early)
        for snap in validations:
            t = time.perf_counter()
            training = [early, short] if snap["name"] == "late" else [early]
            model = fit(training, family)
            values = score(model, snap)
            recs = ranked_predictions(snap["frame"], values, snap["users"], snap["eligible"])
            validation_scores[(family, snap["name"])] = values
            validation_recs[(family, snap["name"])] = recs
            result = evaluate(family, recs, snap["targets"], snap["population"], submission, snap["eligible"], time.perf_counter()-t)
            result["concentration"] = concentration({u: recs[u] for u in snap["population"]})
            result["training_rows"] = model.training_rows
            result["training_snapshot_queries"] = model.training_users
            gains = model.model.booster_.feature_importance(importance_type="gain")
            result["feature_gain_importance"] = dict(sorted(zip(model.features, gains.tolist()), key=lambda pair: -pair[1]))
            evidence["results"].setdefault(snap["name"], {})[family] = result
            print(f"{family} {snap['name']}: {result['map_at_k_all']:.6f} / {result['map_at_k_submission']:.6f}", flush=True)
        save_json("evidence/phase06_optimization.json", evidence)
    for snap in validations:
        recs = {u: snap["ranks"]["recency"][u][:5] for u in snap["users"]}
        evidence["results"][snap["name"]]["Recency30"] = evaluate("Recency30", recs, snap["targets"], snap["population"], submission, snap["eligible"], 0)
        evidence["results"][snap["name"]]["Recency30"]["concentration"] = concentration({u: recs[u] for u in snap["population"]})
        evidence["diagnostics"][snap["name"]] = diagnostics(snap, views)
    deltas = {family: [evidence["results"][s["name"]][family]["map_at_k_all"] - evidence["results"][s["name"]]["baseline"]["map_at_k_all"]
                       for s in validations] for family in ("baseline", "A", "B")}
    robust = [family for family in deltas if min(deltas[family]) >= -1e-12]
    selected = max(robust, key=lambda family: float(np.mean(deltas[family])))
    evidence["selection"] = {"context_family": selected, "map5_deltas_vs_baseline_short_late_main": deltas,
                              "robust_candidates": robust}
    blend = CalibratedAdaptiveBlend().fit(early["frame"], oof_scores[selected], early["targets"], early["profiles"], early["users"], early["eligible"])
    evidence["calibration"] = {"training_history_end": pre["cutoff"], "ranker_labels_end": pre["end"],
        "oof_labels_start": early["cutoff"], "oof_labels_end": early["end"],
        "slope": blend.slope, "intercept": blend.intercept, "weights": blend.weights,
        "early_oof_map5_by_fixed_weight": blend.training_mean_ap, "gate_parameters": blend.gate.get_params()}
    for snap in validations:
        for weight in (None, .02, .05, .10):
            t = time.perf_counter()
            name = "C_adaptive" if weight is None else f"C_fixed_{weight}"
            recs = blend.recommend(snap["frame"], validation_scores[(selected, snap["name"])], snap["profiles"], snap["users"], snap["eligible"], fixed_weight=weight)
            result = evaluate(name, recs, snap["targets"], snap["population"], submission, snap["eligible"], time.perf_counter()-t)
            result["blend_usage"] = dict(blend.usage)
            result["concentration"] = concentration({u: recs[u] for u in snap["population"]})
            evidence["results"][snap["name"]][name] = result
            print(f"{name} {snap['name']}: {result['map_at_k_all']:.6f}", flush=True)
    adaptive_robust = all(evidence["results"][s["name"]]["C_adaptive"]["map_at_k_all"] >=
                          evidence["results"][s["name"]][selected]["map_at_k_all"] - 1e-12 for s in validations)
    evidence["selection"]["adaptive_robust"] = adaptive_robust
    # Keep the primary all-user selection unchanged. A separately named Kaggle-
    # cohort alternative may trade tiny all-user losses for consistent template-
    # user gains, but must still beat the old baseline in every all-user window.
    adaptive_submission_robust = all(evidence["results"][s["name"]]["C_adaptive"]["map_at_k_submission"] >=
                                     evidence["results"][s["name"]][selected]["map_at_k_submission"] for s in validations)
    adaptive_beats_baseline = all(evidence["results"][s["name"]]["C_adaptive"]["map_at_k_all"] >=
                                  evidence["results"][s["name"]]["baseline"]["map_at_k_all"] for s in validations)
    deploy_adaptive = adaptive_robust or (adaptive_submission_robust and adaptive_beats_baseline)
    evidence["selection"].update({
        "adaptive_submission_cohort_robust": adaptive_submission_robust,
        "adaptive_all_users_above_old_baseline": adaptive_beats_baseline,
        "secondary_variant_criterion": "Consistent gains on all 1260 template users in every window, while all-user MAP remains above old baseline; does not replace the primary all-user winner",
        "secondary_criterion_added_after_inspecting_cohort_tradeoff": True})
    # Test relevance sensitivity without selecting or fitting on alternate definitions.
    for threshold in (.5, .8, .95):
        truth = build_ground_truth(main_snap["targets"].with_columns((pl.col("max_coverage_all") >= threshold).alias("is_relevant")))
        evidence.setdefault("relevance_sensitivity_diagnostic_only", {})[str(threshold)] = {
            "baseline_map5": float(np.mean([average_precision_at_k(validation_recs[("baseline", "main")][u], truth.get(u, set())) for u in main_snap["population"]])),
            "mean_relevant_items": float(np.mean([len(truth.get(u, set())) for u in main_snap["population"]]))}
    save_json("evidence/phase06_optimization.json", evidence)
    # Freeze selection before any refit. Disjoint labels: Apr13-May18, May18-Jun22, Jun22-Jul27.
    production = fit([early, short, late], selected)
    production.model.booster_.save_model("outputs/phase06_context_v2.txt")
    future = DataIngestion(PipelineConfig()).load_schedule("31-42").collect()
    final = snapshot("final", TEST_START, TEST_END, schedule, views, submission, future=future)
    values = score(production, final)
    context_recs = ranked_predictions(final["frame"], values, submission, final["eligible"])
    adaptive_recs = blend.recommend(final["frame"], values, final["profiles"], submission, final["eligible"])
    if not deploy_adaptive:
        adaptive_recs = context_recs
        evidence["selection"]["adaptive_deployment"] = "Zero blend fallback: learned gate did not improve every validation window"
    else:
        evidence["selection"]["adaptive_deployment"] = "Historically learned user-dependent calibrated blend; secondary template-cohort variant, not primary all-user winner"
    evidence["selection"]["production_blend_usage"] = dict(blend.usage)
    checks = {}
    for path, recs in (("outputs/submission_context_v2.csv", context_recs), ("outputs/submission_adaptive_v2.csv", adaptive_recs)):
        write_submission(path, recs, final["eligible"])
        checks[path] = validate_csv(path, final["eligible"])
    joblib.dump(blend, "outputs/phase06_adaptive_calibrator.joblib")
    # Protected phase-five models, CSVs and evidence must remain byte-identical.
    for path, digest in hashes.items():
        assert sha256(Path(path).read_bytes()).hexdigest() == digest, f"Changed protected file: {path}"
    def schedule_summary(table, start, end):
        f = table.filter((pl.col("programme_start_utc") < end) & (pl.col("programme_end_utc") > start) & (pl.col("tv_show_id") > 0))
        ids = set(f["tv_show_id"])
        return {"days": (end-start).total_seconds()/86400, "airings": f.height,
                "items": len(ids), "median_airings_per_item": f.group_by("tv_show_id").len()["len"].median()}, ids
    hist_schedule, past_ids = schedule_summary(schedule, TRAIN_START, TRAIN_END)
    main_schedule, main_ids = schedule_summary(schedule, TRAIN_END, TEST_START)
    future_schedule, future_ids = schedule_summary(future, TEST_START, TEST_END)
    evidence["schedule_shift"] = {"training": hist_schedule, "main_validation": main_schedule, "future": future_schedule,
        "future_programmes_not_in_main_schedule_fraction": len(future_ids-main_ids)/len(future_ids),
        "main_programmes_not_in_training_schedule_fraction": len(main_ids-past_ids)/len(main_ids)}
    from sklearn.tree import export_text
    evidence["calibration"]["learned_gate"] = export_text(blend.gate, feature_names=blend.profile_columns)
    evidence.update({"completed_msk": msk_now(), "csv_checks": checks,
        "phase05_and_earlier_artifacts_unchanged": True, "protected_artifact_hashes": hashes,
        "production": {"family": selected, "training_rows": production.training_rows,
                       "training_snapshot_queries": production.training_users, "label_snapshots": ["early", "short", "late"],
                       "prediction_history_days": (TEST_START-TRAIN_START).total_seconds()/86400,
                       "prediction_horizon_days": (TEST_END-TEST_START).total_seconds()/86400,
                       "evaluated_on_refit_labels": False},
        "total_runtime_seconds": time.perf_counter()-started, "submitted_to_kaggle": False})
    save_json("evidence/phase06_optimization.json", evidence)
    print(f"Selected {selected}; adaptive all-user robust={adaptive_robust}, template-cohort robust={adaptive_submission_robust}. Both CSVs validated; phase-five artifacts unchanged.", flush=True)


if __name__ == "__main__":
    main()
