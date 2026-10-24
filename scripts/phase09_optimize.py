"""Fixed dual-ranker experiment, positive retrieval and reproducible v6 exports."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gc
import json
import time
from datetime import timedelta
from hashlib import sha256

import lightgbm as lgb
import numpy as np
import polars as pl

from scripts.phase06_optimize import snapshot, interactions, date, concentration
from scripts.phase08_align import pools_from_frame
from src.constants import TRAIN_START, TRAIN_END, VAL_END, TEST_START, TEST_END, PipelineConfig
from src.contextual_reranking import exposure_features, context_features, ranked_predictions
from src.dual_rankers import PARAMETERS, specialist_features, channel_hour_gate, margin_probabilities, fuse_probabilities
from src.interaction_pipeline import DataIngestion
from src.objective_alignment import airing_window, strength_features, retrieval, top_five_target, dual_metrics, anchor_scores, blend_scores
from src.optimization import historical_schedule, historical_viewing, raw_users, EARLY_CUTOFF, save_json, msk_now
from src.reranking import candidate_features, TemporalRanker
from src.submission import template_users, write_submission, validate_csv

OUT = "evidence/phase09_final_optimization.json"


def corrected_features(snap, views, schedule):
    name, cutoff = snap["name"], snap["cutoff"]
    path = Path(f"data/processed/phase09_{name}_features.parquet")
    info = path.with_suffix(".json")
    gate_path = path.with_name(f"phase09_{name}_gates.parquet")
    if path.exists() and info.exists() and gate_path.exists():
        meta = json.loads(info.read_text())
        snap.update(frame=pl.read_parquet(path), feature_sets=meta["feature_sets"], retrieval=meta["retrieval"], gates=pl.read_parquet(gate_path))
        snap["pool"] = pools_from_frame(snap["frame"])
        return
    started = time.perf_counter()
    legacy_path = Path(f"data/processed/phase08_{name}_features.parquet")
    legacy = pl.read_parquet(legacy_path)
    features = json.loads(legacy_path.with_suffix(".json").read_text())["features"]
    snap["old_features"] = features
    segments = pl.scan_parquet("data/processed/phase08_segments.parquet")
    airings = airing_window(segments, TRAIN_START, cutoff)
    strength = strength_features(airings, cutoff)
    past = schedule.filter((pl.col("programme_start_utc") < cutoff) & (pl.col("programme_end_utc") > TRAIN_START)).unique(
        subset=["channel_id", "programme_start_utc", "tv_show_id"])
    t = time.perf_counter()
    pool, scores, checks = retrieval(airings, strength, views, past, snap["schedule"], cutoff, snap["users"], snap["eligible"], snap["ranks"])
    checks["retrieval_seconds"] = time.perf_counter()-t
    pairs = scores.select("user_id", "tv_show_id")
    source_cols = [c for c in features if c.startswith("score_") or c.startswith("normalized_score_")]
    base_cols = [c for c in features if c not in source_cols and c != "rank_pool"]
    common = legacy.join(pairs, on=["user_id", "tv_show_id"], how="inner").select(["user_id", "tv_show_id"]+base_cols)
    missing = pairs.join(legacy.select("user_id", "tv_show_id"), on=["user_id", "tv_show_id"], how="anti")
    if missing.height:
        missing_pool = pools_from_frame(missing)
        ranks = {k: snap["ranks"][k] for k in ("global", "frequency", "recency", "ppr", "cbf")}
        ranks["pool"] = missing_pool
        extra, _ = candidate_features(snap["history"], ranks, snap["users"], snap["eligible"], cutoff, snap["schedule"], depth=220)
        extra = extra.join(missing, on=["user_id", "tv_show_id"], how="inner")
        recent = {d: interactions(max(TRAIN_START, cutoff-timedelta(days=d)), cutoff) for d in (7, 14, 28)}
        extra, _, _ = exposure_features(extra, snap["history"], recent, views, schedule, snap["schedule"], TRAIN_START, cutoff, snap["end"])
        extra, _ = context_features(extra, recent, views, schedule, snap["schedule"], TRAIN_START, cutoff)
        extra = extra.join(strength, on=["user_id", "tv_show_id"], how="left").with_columns(pl.col("observed_days_ago").fill_null(10000)).fill_null(0)
        sf = [c for c in strength.columns if c not in ("user_id", "tv_show_id", "observed_days_ago")]
        extra = extra.with_columns([pl.col(c).cast(pl.Float64).log1p() for c in sf])
        extra = extra.select(["user_id", "tv_show_id"]+[pl.col(c).cast(pl.Float32) for c in base_cols])
        common = pl.concat([common, extra])
    ranks = pl.DataFrame([(u, p, r) for u, values in pool.items() for r, p in enumerate(values, 1)],
                         schema=[("user_id", pl.String), ("tv_show_id", pl.Int64), ("rank_pool", pl.Float32)], orient="row")
    frame = common.join(scores, on=["user_id", "tv_show_id"], how="inner").join(ranks, on=["user_id", "tv_show_id"], how="inner")
    raw_score_cols = [c for c in scores.columns if c.startswith("score_")]
    frame = frame.with_columns([(pl.col(c).log1p()/pl.col(c).log1p().max().over("user_id").clip(1e-12)).alias("normalized_"+c) for c in raw_score_cols])
    frame = frame.select(["user_id", "tv_show_id"]+[pl.col(c).cast(pl.Float32) for c in features]).sort(["user_id", "tv_show_id"])
    frame, sets = specialist_features(frame, views, snap["schedule"], cutoff, snap["end"])
    gates = channel_hour_gate(views, snap["users"], cutoff)
    sets["V5_reference"] = features
    assert np.isfinite(frame.select(list(dict.fromkeys(sum(sets.values(), [])))).to_numpy()).all()
    assert not frame.select("user_id", "tv_show_id").is_duplicated().any()
    assert all(len(v) == min(220, len(snap["eligible"])) for v in pool.values())
    checks.update(feature_seconds_total=time.perf_counter()-started, reused_phase08_rows=common.height-missing.height,
                  new_feature_rows=missing.height, rows=frame.height)
    frame.write_parquet(path); gates.write_parquet(gate_path)
    save_json(info, {"feature_sets": sets, "retrieval": checks, "cutoff": cutoff, "target_end": snap["end"]})
    snap.update(frame=frame, feature_sets=sets, retrieval=checks, gates=gates, pool=pool)
    print(f"features {name}: {frame.height} rows, {missing.height} new rows, {checks['feature_seconds_total']:.1f}s", flush=True)


def fit(snaps, family, reference=False):
    if reference:
        features = json.loads(Path("data/processed/phase08_early_features.json").read_text())["features"]
    else:
        features = snaps[0]["feature_sets"][family]
    frames, labels = [], []
    for snap in snaps:
        f = pl.read_parquet(f"data/processed/phase08_{snap['name']}_features.parquet").select(["user_id", "tv_show_id"]+features) if reference else snap["frame"].select(["user_id", "tv_show_id"]+features)
        prefix = snap["name"]+":"
        frames.append(f.with_columns((pl.lit(prefix)+pl.col("user_id")).alias("user_id")))
        labels.append(snap["top_target"].select("user_id", "tv_show_id", "is_relevant").with_columns((pl.lit(prefix)+pl.col("user_id")).alias("user_id")))
    started = time.perf_counter()
    model = TemporalRanker(None if reference else PARAMETERS).fit(pl.concat(frames), features, pl.concat(labels))
    return model, time.perf_counter()-started


def score(model, frame, features):
    return model.predict(frame.select(features).to_numpy().astype(np.float32), num_threads=4)


def variants(snap, recent, horizon, old):
    pa, pb = margin_probabilities(snap["frame"], recent), margin_probabilities(snap["frame"], horizon)
    equal = fuse_probabilities(snap["frame"], pa, pb)
    stable = fuse_probabilities(snap["frame"], pa, pb, snap["gates"])
    scores = {"A_recent": pa, "B_horizon": pb, "D_equal": equal, "E_stability": stable, "V5_corrected": old}
    for name, ensemble in (("D_equal", equal), ("E_stability", stable)):
        for weight in (.25, .50, .75):
            scores[f"{name}_v5_{weight}"] = weight*ensemble+(1-weight)*old
    return scores


def main():
    started = time.perf_counter()
    old_evidence = json.loads(Path("evidence/phase08_target_alignment.json").read_text())
    protected = {str(p): sha256(p.read_bytes()).hexdigest() for root in ("outputs", "evidence") for p in Path(root).glob("*")
                 if p.is_file() and not p.name.startswith("phase09") and p.name not in ("submission_dual_v6.csv", "submission_stability_v6.csv")}
    evidence = {"created_msk": msk_now(), "methodology": {
        "target": "Reuse independently verified >=80% distinct-airing Top5 counts; numeric ID tie-break; empty users retained",
        "cohorts": {"all": 4876, "submission": 1260}, "primary_metric": "Submission-cohort Top5 frequency MAP5",
        "parameters_both_specialists": PARAMETERS, "softmax": "exp(margin-user_max_margin), divided by user sum",
        "stability_hypothesis": "recent weight=clip(2*(cos(channel-hour last7,last28)-0.5),0,1); no future labels",
        "reference": "Original v5 rescored with original cached candidates; strict historical refits of its unchanged configuration. Reference on corrected candidates used for fusion.",
        "selection": "Prefer no loss versus original v5 on either disjoint window; otherwise allow <=0.001 absolute MAP loss per window; maximize two-window mean. Main is overlapping diagnostic only.",
        "mix_weights": [.25, .5, .75], "mix_definition": "weight*dual_ranker_probability+(1-weight)*v5_score",
        "optional_weekly_replay": "Not attempted: bounded dual-ranker experiment prioritized",
        "kaggle_user_reported_diagnostic_only": {"objective_v5_public": .45154, "objective_v5_private": .43744, "context_v2_private": .43051, "anchor_v5_private": .41992}},
        "results": {}, "retrieval": {}, "timing": {}, "reference_reproduction": {}, "windows": {}}
    views = historical_viewing().collect(); schedule = historical_schedule().collect()
    users = template_users(); population = raw_users(TRAIN_START, TRAIN_END)
    populations = {"all": population, "submission": users}
    early = snapshot("early", EARLY_CUTOFF, TRAIN_END, schedule, views, users)
    short = snapshot("short", TRAIN_END, date(6, 22), schedule, views, users)
    late = snapshot("late", date(6, 22), VAL_END, schedule, views, users)
    main_snap = snapshot("main", TRAIN_END, VAL_END, schedule, views, users)
    snaps = [early, short, late, main_snap]
    for snap in snaps:
        snap["all_target"] = snap["targets"]
        snap["top_target"] = top_five_target(snap["targets"])
        snap["schedule"] = schedule.filter((pl.col("programme_start_utc") < snap["end"]) & (pl.col("programme_end_utc") > snap["cutoff"]) & (pl.col("tv_show_id") > 0)).unique(subset=["channel_id", "programme_start_utc", "tv_show_id"])
        snap.pop("frame", None); snap.pop("profiles", None)
        corrected_features(snap, views, schedule)
        evidence["retrieval"][snap["name"]] = snap["retrieval"]
        evidence["windows"][snap["name"]] = {"cutoff": snap["cutoff"], "target_end": snap["end"]}
    evidence["feature_sets"] = early["feature_sets"]
    reference_models, specialists = {}, {}
    for label, training in (("early", [early]), ("late", [early, short])):
        model, seconds = fit(training, "V5_reference", reference=True)
        reference_models[label] = model
        evidence["timing"][f"reference_fit_{label}"] = seconds
        for family in ("A_recent", "B_horizon"):
            model, seconds = fit(training, family)
            specialists[(label, family)] = model
            evidence["timing"][f"{family}_fit_{label}"] = seconds
            print(f"fit {label} {family}: {seconds:.1f}s", flush=True)
    for snap in (short, late, main_snap):
        name = snap["name"]; label = "late" if name == "late" else "early"
        assert max(s["end"] for s in ([early, short] if label == "late" else [early])) <= snap["cutoff"]
        ref = reference_models[label]
        legacy = pl.read_parquet(f"data/processed/phase08_{name}_features.parquet")
        old_margin = score(ref.model.booster_, legacy, ref.features)
        original_scores = blend_scores(legacy, old_margin, anchor_scores(legacy, 10, "balanced"), .2, "rank")
        original_recs = ranked_predictions(legacy, original_scores, snap["users"], snap["eligible"])
        old_pool = pools_from_frame(legacy)
        result = {c: dual_metrics(original_recs, snap["all_target"], snap["top_target"], ids, old_pool) for c, ids in populations.items()}
        expected = old_evidence["results"][name]["B_rank_0.2"]["submission"]["top5_frequency"]["map5"]
        actual = result["submission"]["top5_frequency"]["map5"]
        assert abs(expected-actual) < 1e-12, (name, expected, actual)
        evidence["reference_reproduction"][name] = {"matches_phase08": True, "map5": actual}
        result["concentration"] = {c: concentration({u: original_recs[u] for u in ids}) for c, ids in populations.items()}
        evidence["results"][name] = {"V5_original": result}
        begin = time.perf_counter()
        frame = snap["frame"]
        old = blend_scores(frame, score(ref.model.booster_, frame, ref.features), anchor_scores(frame, 10, "balanced"), .2, "rank")
        margins = {family: score(specialists[(label, family)].model.booster_, frame, snap["feature_sets"][family]) for family in ("A_recent", "B_horizon")}
        prediction_seconds = time.perf_counter()-begin
        for variant, scores in variants(snap, margins["A_recent"], margins["B_horizon"], old).items():
            begin = time.perf_counter()
            recs = ranked_predictions(frame, scores, snap["users"], snap["eligible"])
            result = {c: dual_metrics(recs, snap["all_target"], snap["top_target"], ids, snap["pool"]) for c, ids in populations.items()}
            result["concentration"] = {c: concentration({u: recs[u] for u in ids}) for c, ids in populations.items()}
            result["ranking_evaluation_seconds"] = time.perf_counter()-begin
            result["shared_all_model_prediction_seconds"] = prediction_seconds
            evidence["results"][name][variant] = result
        for variant in ("V5_original", "V5_corrected", "A_recent", "B_horizon", "D_equal", "E_stability"):
            print(f"{name} {variant}: {evidence['results'][name][variant]['submission']['top5_frequency']['map5']:.6f}", flush=True)
        save_json(OUT, evidence)
    def metric(window, name):
        return evidence["results"][window][name]["submission"]["top5_frequency"]["map5"]
    def choose(names, fallback):
        good = [n for n in names if all(metric(w,n) >= metric(w,"V5_original") for w in ("short","late"))]
        acceptable = good or [n for n in names if all(metric(w,n) >= metric(w,"V5_original")-.001 for w in ("short","late"))]
        return max(acceptable, key=lambda n: (np.mean([metric(w,n) for w in ("short","late")]), n)) if acceptable else fallback
    dual_names = ["D_equal"]+[f"D_equal_v5_{w}" for w in (.25,.5,.75)]
    stable_names = ["E_stability"]+[f"E_stability_v5_{w}" for w in (.25,.5,.75)]
    dual = choose(dual_names, "V5_original")
    stability = choose(stable_names, dual)
    evidence["selection"] = {"dual": dual, "stability": stability,
        "substantial_regression_threshold": .001, "main_not_used_for_selection": True}
    save_json(OUT, evidence)
    del main_snap["frame"]
    reference_models.clear(); specialists.clear(); gc.collect()
    production = {}
    for family in ("A_recent", "B_horizon"):
        model, seconds = fit([early,short,late], family)
        model.model.booster_.save_model(f"outputs/phase09_{family}.txt")
        production[family] = model
        evidence["timing"][family+"_production_refit"] = seconds
    future = DataIngestion(PipelineConfig()).load_schedule("31-42").collect()
    final = snapshot("final", TEST_START, TEST_END, schedule, views, users, future=future)
    final["schedule"] = future.filter((pl.col("programme_start_utc") < TEST_END) & (pl.col("programme_end_utc") > TEST_START) & (pl.col("tv_show_id") > 0)).unique(subset=["channel_id", "programme_start_utc", "tv_show_id"])
    final.pop("frame", None); final.pop("profiles", None)
    corrected_features(final, views, schedule)
    evidence["retrieval"]["final"] = final["retrieval"]
    legacy_model = lgb.Booster(model_file="outputs/phase08_objective_v5.txt")
    legacy_features = json.loads(Path("data/processed/phase08_final_features.json").read_text())["features"]
    frame = final["frame"]
    old = blend_scores(frame, score(legacy_model,frame,legacy_features), anchor_scores(frame,10,"balanced"), .2,"rank")
    margins = {family: score(model.model.booster_,frame,model.features) for family,model in production.items()}
    values = variants(final,margins["A_recent"],margins["B_horizon"],old)
    checks = {}
    for path, selected in (("outputs/submission_dual_v6.csv",dual),("outputs/submission_stability_v6.csv",stability)):
        if selected == "V5_original":
            legacy = pl.read_parquet("data/processed/phase08_final_features.parquet")
            pred = ranked_predictions(legacy,blend_scores(legacy,score(legacy_model,legacy,legacy_features),anchor_scores(legacy,10,"balanced"),.2,"rank"),users,final["eligible"])
        else:
            pred = ranked_predictions(frame,values[selected],users,final["eligible"])
            assert pred == ranked_predictions(frame,variants(final,margins["A_recent"],margins["B_horizon"],old)[selected],users,final["eligible"])
        write_submission(path,pred,final["eligible"])
        checks[path] = validate_csv(path,final["eligible"])
    for path,digest in protected.items(): assert sha256(Path(path).read_bytes()).hexdigest()==digest,path
    evidence.update({"completed_msk":msk_now(),"csv_checks":checks,"protected_artifacts_unchanged":True,"protected_sha256":protected,
        "candidate_alignment": "Both rankers predict the identical sorted corrected-candidate frame; every probability vector retains its row order",
        "production": {"label_windows":[[s["cutoff"],s["end"]] for s in (early,short,late)],"history_cutoff":TEST_START,
            "future_end":TEST_END,"deterministic":True,"evaluated_on_training_labels":False},
        "code_sha256":{p:sha256(Path(p).read_bytes()).hexdigest() for p in ("src/objective_alignment.py","src/reranking.py","src/dual_rankers.py","scripts/phase09_optimize.py")},
        "runtime_seconds":time.perf_counter()-started,"submitted_to_kaggle":False})
    save_json(OUT,evidence)
    print(f"Selected dual={dual}, stability={stability}; both CSVs validated; previous artifacts unchanged.",flush=True)


if __name__ == "__main__":
    main()
