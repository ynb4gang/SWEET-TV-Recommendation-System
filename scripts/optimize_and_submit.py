"""Reproducible local optimization and final CSVs. Never accesses Kaggle."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import importlib.metadata
import json
import time
import hashlib

import numpy as np
import polars as pl

from src.constants import TRAIN_START, TRAIN_END, VAL_END, TEST_START, TEST_END, PipelineConfig
from src.interaction_pipeline import DataIngestion
from src.baselines import GlobalPopularityRecommender, PersonalizedFrequencyRecommender, RecencyWeightedRecommender, get_schedule_eligible_programmes
from src.content_based import ContentBasedFilteringRecommender
from src.personalized_pagerank import PersonalizedPageRankRecommender
from src.optimization import (window_interactions, historical_schedule, historical_viewing, raw_users,
                              populations, evaluate, rrf, save_json, msk_now, EARLY_CUTOFF)
from src.reranking import AdaptiveHybrid, TemporalRanker, candidate_features
from src.submission import template_users, write_submission, validate_predictions

DEPTH = 100
VERSION = "phase05_verified_v1"


def component(name, model, history, schedule, eligible, users, window):
    cache = Path(f"data/processed/{VERSION}_{window}_{name}.parquet")
    info = cache.with_suffix(".json")
    if cache.exists() and info.exists():
        frame = pl.read_parquet(cache)
        return dict(frame.iter_rows()), json.loads(info.read_text())
    start = time.perf_counter()
    if name == "cbf":
        model.fit(history, schedule.lazy(), eligible)
    else:
        model.fit(history, eligible)
    recs = model.recommend_batch(users, DEPTH)
    if recs != model.recommend_batch(users, DEPTH):
        raise AssertionError(f"Nondeterministic {name}")
    # All prefixes are independently checked, including cold-start users.
    validate_predictions({u: r[:5] for u, r in recs.items()}, users, eligible)
    runtime = time.perf_counter()-start
    stats = {"runtime_seconds": runtime, "determinism_passed": True}
    if hasattr(model, "convergence"):
        stats["convergence"] = model.convergence
        if model.convergence["unconverged_users"]:
            raise RuntimeError(f"{name} failed convergence: {model.convergence}")
    if hasattr(model, "user_rankings"):
        stats["fallback_users"] = sum(len(model.user_rankings.get(u, [])) < 5 for u in users)
    pl.DataFrame({"user_id": users, "ranking": [recs[u] for u in users]}).write_parquet(cache)
    save_json(info, stats)
    print(f"{window}: {name} finished ({runtime:.2f}s)", flush=True)
    return recs, stats


def components(history, schedule, eligible, users, cutoff, window, ppr_alphas=(0.15, 0.30)):
    models = {
        "global": GlobalPopularityRecommender("watched_seconds"),
        "frequency": PersonalizedFrequencyRecommender(),
        "recency": RecencyWeightedRecommender(half_life_days=30, cutoff=cutoff),
        "cbf": ContentBasedFilteringRecommender(profile_weighting="coverage"),
    }
    for alpha in ppr_alphas:
        models[f"ppr_{alpha}"] = PersonalizedPageRankRecommender(alpha=alpha, max_iter=200, tol=1e-6)
    rankings, stats = {}, {}
    for name, model in models.items():
        rankings[name], stats[name] = component(name, model, history, schedule, eligible, users, window)
    return rankings, stats


HYBRIDS = {
    "R80_P20": (["recency", "ppr"], [0.8, 0.2]),
    "R60_P40": (["recency", "ppr"], [0.6, 0.4]),
    "R40_P60": (["recency", "ppr"], [0.4, 0.6]),
    "F20_R40_P40": (["frequency", "recency", "ppr"], [0.2, 0.4, 0.4]),
    "F40_R30_P30": (["frequency", "recency", "ppr"], [0.4, 0.3, 0.3]),
    "R50_P40_G10": (["recency", "ppr", "global"], [0.5, 0.4, 0.1]),
    "R40_P40_G20": (["recency", "ppr", "global"], [0.4, 0.4, 0.2]),
    "R50_P40_C10": (["recency", "ppr", "cbf"], [0.5, 0.4, 0.1]),
    "R40_P40_C20": (["recency", "ppr", "cbf"], [0.4, 0.4, 0.2]),
}


def main():
    start = time.perf_counter()
    # Previously saved evidence is not overwritten. Its limitations are explicit.
    save_json("evidence/phase05_prior_results_status.json", {
        "phase05_ppr.json": "Historical result: invalid positive-only claim (partial edges admitted); superseded, preserved",
        "phase03_baselines.json": "INVALID: target-derived candidates and positive-only population; preserved",
        "phase03_1_corrected_baselines.json": "Historical restricted cohort and omitted cold-start predictions; preserved",
        "phase04_cbf.json": "Historical cohort excludes raw users without recognized programmes; preserved"})
    train = window_interactions("train", TRAIN_START, TRAIN_END)
    val = window_interactions("validation", TRAIN_END, VAL_END)
    a, b = populations()
    users = list(dict.fromkeys(a+b))
    sched = historical_schedule().collect()
    val_sched = sched.filter((pl.col("programme_start_utc") < VAL_END) & (pl.col("programme_end_utc") > TRAIN_END))
    eligible = get_schedule_eligible_programmes(val_sched.lazy(), TRAIN_END, VAL_END)
    ranks, timing = components(train, sched, eligible, users, TRAIN_END, "validation")
    results, predictions = {}, {}
    for name, recs in ranks.items():
        predictions[name] = {u: r[:5] for u, r in recs.items()}
        results[name] = evaluate(name, predictions[name], val, a, b, eligible, timing[name]["runtime_seconds"])
        results[name].update({k: v for k, v in timing[name].items() if k != "runtime_seconds"})
    best_ppr = max((n for n in ranks if n.startswith("ppr_")), key=lambda n: results[n]["map_at_k_all"])
    ranks["ppr"] = ranks[best_ppr]
    for name, (keys, weights) in HYBRIDS.items():
        t = time.perf_counter()
        predictions[name] = rrf([ranks[k] for k in keys], weights, users, eligible)
        runtime = time.perf_counter()-t + sum(timing[best_ppr if k == "ppr" else k]["runtime_seconds"] for k in keys)
        results[name] = evaluate(name, predictions[name], val, a, b, eligible, runtime)
        results[name]["weights"] = dict(zip(keys, weights))
    evidence = {"created_msk": msk_now(), "metric": "MAP@5 binary, denominator=min(5,|relevant|); local proxy",
                "population_a": "All raw historical viewing users before training cutoff",
                "population_b": "All template users, original order",
                "selection_metric": "map_at_k_all", "eligible_programmes": len(eligible),
                "cohorts": {"all": len(a), "submission": len(b)}, "results": results,
                "recency": "aggregate last-interaction proxy, not per-airing exponential decay"}
    save_json("evidence/phase05_optimization.json", evidence)

    # Early fitting snapshot ends at the established validation START. It never
    # contains May18--July26 targets, for either gating or supervised fitting.
    early_train = window_interactions("early_train", TRAIN_START, EARLY_CUTOFF)
    early_targets = window_interactions("early_targets", EARLY_CUTOFF, TRAIN_END)
    early_users = raw_users(TRAIN_START, EARLY_CUTOFF)
    early_sched = sched.filter((pl.col("programme_start_utc") < TRAIN_END) &
                              (pl.col("programme_end_utc") > EARLY_CUTOFF))
    early_eligible = get_schedule_eligible_programmes(early_sched.lazy(), EARLY_CUTOFF, TRAIN_END)
    # Item metadata seen at this snapshot excludes schedules after its target end.
    early_metadata = sched.filter(pl.col("programme_start_utc") < TRAIN_END)
    alpha = float(best_ppr.split("_")[1])
    eranks, etiming = components(early_train, early_metadata, early_eligible, early_users,
                                EARLY_CUTOFF, "early", (alpha,))
    eranks["ppr"] = eranks[best_ppr]
    gate = AdaptiveHybrid().fit(early_train, early_targets, early_users, eranks, early_eligible)
    t = time.perf_counter()
    predictions["adaptive"] = gate.recommend(train, ranks, users, eligible)
    results["adaptive"] = evaluate("adaptive", predictions["adaptive"], val, a, b, eligible,
                                    time.perf_counter()-t + timing["recency"]["runtime_seconds"] + timing[best_ppr]["runtime_seconds"])
    results["adaptive"]["gate_usage"] = gate.usage
    evidence["supervised_windows"] = {"history_end_utc": EARLY_CUTOFF,
                                      "labels_end_exclusive_utc": TRAIN_END,
                                      "validation_start_utc": TRAIN_END}
    # Train one fixed, small LambdaRank configuration; no final-period early stopping.
    ranker = None
    try:
        t = time.perf_counter()
        keys = ["global", "frequency", "recency", "ppr", "cbf"]
        early_frame, features = candidate_features(early_train, {k: eranks[k] for k in keys},
                                                  early_users, early_eligible, EARLY_CUTOFF, early_sched)
        ranker = TemporalRanker().fit(early_frame, features, early_targets)
        fit_seconds = time.perf_counter()-t
        t = time.perf_counter()
        val_frame, _ = candidate_features(train, {k: ranks[k] for k in keys}, users, eligible, TRAIN_END, val_sched)
        predictions["lightgbm"] = ranker.recommend(val_frame, users, eligible)
        results["lightgbm"] = evaluate("lightgbm", predictions["lightgbm"], val, a, b, eligible,
                                      fit_seconds + time.perf_counter()-t + sum(timing[best_ppr if k=="ppr" else k]["runtime_seconds"] for k in keys))
        results["lightgbm"].update({"training_rows": ranker.training_rows, "training_users": ranker.training_users,
                                   "features": features, "fit_seconds": fit_seconds, "parameters": ranker.model.get_params()})
        # Normalized rank fusion also checks whether ranker's incremental signal helps.
        for weight in (0.2, 0.5, 0.8):
            name = f"LGB{weight}_PPR{1-weight:.1f}"
            t = time.perf_counter()
            predictions[name] = rrf([predictions["lightgbm"], ranks["ppr"]], [weight, 1-weight], users, eligible)
            results[name] = evaluate(name, predictions[name], val, a, b, eligible,
                                    time.perf_counter()-t+results["lightgbm"]["runtime_seconds"])
        Path("outputs").mkdir(exist_ok=True)
        ranker.model.booster_.save_model("outputs/phase05_ranker.txt")
    except (ImportError, OSError) as error:
        evidence["lightgbm_skipped"] = f"{type(error).__name__}: {error}"
        print(evidence["lightgbm_skipped"], flush=True)
    best = max(results, key=lambda n: results[n]["map_at_k_all"])
    evidence.update({"winner": best, "best_ppr": best_ppr, "results": results,
                     "elapsed_seconds_before_final": time.perf_counter()-start})
    save_json("evidence/phase05_optimization.json", evidence)
    save_json("evidence/phase05_verified_ppr.json", {n: results[n] for n in results if n.startswith("ppr_")})
    save_json("evidence/phase04_verified_baselines.json", {n: results[n] for n in ("global", "frequency", "recency")})
    print(f"Validation winner: {best} = {results[best]['map_at_k_all']:.6f}", flush=True)

    final_refit = {"performed": False}
    if best == "lightgbm" or best.startswith("LGB"):
        # Selection is now frozen. Use all available labelled historical snapshots
        # for the production model only. Never evaluate this refit on its own labels.
        t = time.perf_counter()
        early_labels = early_targets.filter(pl.col("is_relevant")).select("user_id", "tv_show_id")
        val_labels = val.filter(pl.col("is_relevant")).select("user_id", "tv_show_id")
        # Snapshot-specific labels must stay attached to each row: the same pair
        # can be relevant in one snapshot but not the other.
        from src.reranking import fit_production_ranker
        ranker = fit_production_ranker(early_frame, early_labels, val_frame, val_labels, features)
        ranker.model.booster_.save_model("outputs/phase05_ranker_full_history.txt")
        final_refit = {"performed": True, "runtime_seconds": time.perf_counter()-t,
                       "training_rows": ranker.training_rows, "training_users": ranker.training_users,
                       "label_windows": [[EARLY_CUTOFF, TRAIN_END], [TRAIN_END, TEST_START]],
                       "evaluated_on_training_labels": False,
                       "selection_frozen_before_refit": True}

    # Refit all components on the true full historical window, not concatenated
    # per-programme aggregates. No future viewing exists or is read.
    full = window_interactions("full_history", TRAIN_START, TEST_START)
    future = DataIngestion(PipelineConfig()).load_schedule("31-42").collect()
    future_window = future.filter((pl.col("programme_start_utc") < TEST_END) & (pl.col("programme_end_utc") > TEST_START))
    final_eligible = get_schedule_eligible_programmes(future_window.lazy(), TEST_START, TEST_END)
    final_metadata = pl.concat([sched, future])
    final_ranks, final_timing = components(full, final_metadata, final_eligible, b, TEST_START, "final", (alpha,))
    final_ranks["ppr"] = final_ranks[best_ppr]
    backup = {u: final_ranks["recency"][u][:5] for u in b}
    backup_check = write_submission("outputs/submission_recency30_backup.csv", backup, final_eligible)
    if best in final_ranks:
        final = {u: final_ranks[best][u][:5] for u in b}
    elif best in HYBRIDS:
        keys, weights = HYBRIDS[best]
        final = rrf([final_ranks[k] for k in keys], weights, b, final_eligible)
    elif best == "adaptive":
        final = gate.recommend(full, final_ranks, b, final_eligible)
    elif best == "lightgbm" or best.startswith("LGB"):
        keys = ["global", "frequency", "recency", "ppr", "cbf"]
        final_frame, _ = candidate_features(full, {k: final_ranks[k] for k in keys}, b, final_eligible, TEST_START, future_window)
        final = ranker.recommend(final_frame, b, final_eligible)
        if best.startswith("LGB"):
            weight = float(best.split("_")[0][3:])
            final = rrf([final, final_ranks["ppr"]], [weight, 1-weight], b, final_eligible)
    else:
        raise RuntimeError(f"Unknown winning model: {best}")
    validate_predictions(final, b, final_eligible)
    best_path = Path("outputs/submission_best.csv")
    if best_path.exists():
        # Preserve any CSV generated earlier in this session rather than discard it.
        prior_hash = hashlib.sha256(best_path.read_bytes()).hexdigest()
        prior_path = best_path.with_name(f"submission_best_preserved_{prior_hash[:12]}.csv")
        if not prior_path.exists():
            prior_path.write_bytes(best_path.read_bytes())
        if prior_path.read_bytes() != best_path.read_bytes():
            raise RuntimeError(f"Preserved CSV differs from {best_path}")
        # Only remove our own generated file after its bytes are safely preserved.
        import csv
        with best_path.open(newline="") as f:
            old = list(csv.DictReader(f))
        expected = [{"user_id": u, "tv_show_id": " ".join(map(str, final[u]))} for u in b]
        if old != expected:
            best_path.unlink()
    best_check = write_submission(best_path, final, final_eligible)
    # Actual selected-user recommendations plus historical evidence, never invented explanations.
    examples = []
    for u in b[:2]:
        examples.append({"user_id": u, "recommendations": final[u], "history": full.filter(pl.col("user_id")==u)
                         .sort("total_broadcasts_watched", descending=True).head(10).to_dicts()})
    save_json("evidence/phase05_selected_users.json", examples)
    save_json("evidence/phase05_final_selection.json", {
        "created_msk": msk_now(), "winner": best, "parameters": results[best].get("parameters", results[best].get("weights", {})),
        "validation": results[best], "backup_validation": results["recency"],
        "prediction_cutoff_utc": TEST_START, "training_pairs_full_history": full.height,
        "future_candidates": len(final_eligible), "final_component_timing": final_timing,
        "production_ranker_refit": final_refit,
        "csv_checks": {"outputs/submission_best.csv": best_check, "outputs/submission_recency30_backup.csv": backup_check},
        "total_runtime_seconds": time.perf_counter()-start, "submitted": False,
        "dependencies": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "polars", "scikit-learn", "lightgbm")},
        "limitations": ["Local proxy; hidden Kaggle target unverified", "Main split used for limited model selection; not an untouched test",
                        "Recency uses aggregate timestamps", "Offline ranker/gate fitted before main validation; production ranker refit only after selection"]})
    for name, result in results.items():
        print(f"{name:22} {result['map_at_k_all']:.6f} {result['map_at_k_submission']:.6f} {result['hit_rate_at_k_all']:.6f} {result['runtime_seconds']:.2f}")
    print("Final CSVs validated; no Kaggle submission performed.")


if __name__ == "__main__":
    main()
