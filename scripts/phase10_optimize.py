"""One completion/replay LambdaRank experiment; phase-five--nine artifacts read-only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import gc
import json
import time
from hashlib import sha256

import numpy as np
import polars as pl

from scripts.phase06_optimize import snapshot,interactions,date
from src.completion_replay import (viewing_union,scheduled_parts,exact_context,conditional_completion,
    weekly_replay,personal_anchor,augment_scores,missing_retrieval_scores,ANCHOR_WEIGHTS)
from src.constants import TRAIN_START,TRAIN_END,VAL_END,TEST_START,TEST_END,PipelineConfig
from src.contextual_reranking import exposure_features,context_features,ranked_predictions
from src.dual_rankers import margin_probabilities
from src.interaction_pipeline import DataIngestion
from src.objective_alignment import airing_window,strength_features,top_five_target,dual_metrics
from src.optimization import historical_schedule,historical_viewing,raw_users,EARLY_CUTOFF,save_json,msk_now
from src.reranking import candidate_features,TemporalRanker
from src.submission import template_users,write_submission,validate_csv

OUT="evidence/phase10_final_experiment.json"


def pool_from(frame):
    return {k[0]:g["tv_show_id"].to_list() for k,g in frame.partition_by("user_id",as_dict=True).items()}


def build(snap,views,schedule):
    started=time.perf_counter();name=snap["name"]
    path=Path(f"data/processed/phase10_{name}_features.parquet");info=path.with_suffix(".json")
    if path.exists() and info.exists():
        meta=json.loads(info.read_text());snap.update(frame=pl.read_parquet(path),features=meta["features"],timing=meta["timing"])
        snap["pool"]=pool_from(snap["frame"]);return
    cutoff,end=snap["cutoff"],snap["end"]
    legacy=pl.read_parquet(f"data/processed/phase09_{name}_features.parquet")
    base_features=json.loads(Path(f"data/processed/phase08_{name}_features.json").read_text())["features"]
    legacy=legacy.select(["user_id","tv_show_id"]+base_features)
    airings=airing_window(pl.scan_parquet("data/processed/phase08_segments.parquet"),TRAIN_START,cutoff)
    strength=strength_features(airings,cutoff)
    past_raw=schedule.filter((pl.col("programme_start_utc")<cutoff)&(pl.col("programme_end_utc")>TRAIN_START)&(pl.col("tv_show_id")>0)).unique(
        subset=["channel_id","programme_start_utc","tv_show_id"])
    past=scheduled_parts(past_raw,TRAIN_START,cutoff)
    future=scheduled_parts(snap["schedule"],cutoff,end)
    t=time.perf_counter()
    cc,cc_names,behavior=conditional_completion(legacy.select("user_id","tv_show_id"),airings,past,future,
        snap["users"],snap["eligible"],extra=20)
    timing={"conditional_seconds":time.perf_counter()-t}
    pairs=cc.select("user_id","tv_show_id")
    missing=pairs.join(legacy.select("user_id","tv_show_id"),on=["user_id","tv_show_id"],how="anti")
    if missing.height:
        ranks={k:snap["ranks"][k] for k in ("global","frequency","recency","ppr","cbf")};ranks["pool"]=pool_from(missing)
        extra,_=candidate_features(snap["history"],ranks,snap["users"],snap["eligible"],cutoff,snap["schedule"],depth=240)
        extra=extra.join(missing,on=["user_id","tv_show_id"],how="inner")
        from datetime import timedelta
        recent={d:interactions(max(TRAIN_START,cutoff-timedelta(days=d)),cutoff) for d in (7,14,28)}
        extra,_,_=exposure_features(extra,snap["history"],recent,views,schedule,snap["schedule"],TRAIN_START,cutoff,end)
        extra,_=context_features(extra,recent,views,schedule,snap["schedule"],TRAIN_START,cutoff)
        extra=extra.join(strength,on=["user_id","tv_show_id"],how="left").with_columns(pl.col("observed_days_ago").fill_null(10000)).fill_null(0)
        extra=extra.with_columns([pl.col(c).cast(pl.Float64).log1p() for c in strength.columns if c not in ("user_id","tv_show_id","observed_days_ago")])
        raw=missing_retrieval_scores(missing,strength,airings,past_raw,snap["schedule"],snap["ranks"])
        extra=extra.join(raw,on=["user_id","tv_show_id"],how="left").with_columns(pl.lit(221.).alias("rank_pool"))
        for c in [c for c in raw.columns if c.startswith("score_")]:extra=extra.with_columns(pl.lit(0.).alias("normalized_"+c))
        extra=extra.select(["user_id","tv_show_id"]+[pl.col(c).cast(pl.Float32) for c in base_features])
        legacy=pl.concat([legacy,extra])
    union=viewing_union(views,TRAIN_START,cutoff)
    t=time.perf_counter()
    frame,context_new=exact_context(legacy,union,future,cutoff)
    timing["exact_context_seconds"]=time.perf_counter()-t
    frame=frame.with_columns((.5*pl.col("compatibility_channel_28")+.3*pl.col("compatibility_channel_hour_28")+
        .2*pl.col("compatibility_channel_dow_28")).alias("score_context_retrieval"))
    raw_cols=[c for c in base_features if c.startswith("score_")]
    frame=frame.with_columns([(pl.col(c).log1p()/pl.col(c).log1p().max().over("user_id").clip(1e-12)).alias("normalized_"+c) for c in raw_cols])
    t=time.perf_counter()
    replay=weekly_replay(union,future,pairs,TRAIN_START,cutoff,f"data/processed/phase10_{name}_weekly.parquet")
    timing["weekly_seconds"]=time.perf_counter()-t
    frame=frame.join(cc,on=["user_id","tv_show_id"],how="left").join(replay,on=["user_id","tv_show_id"],how="left").join(
        behavior,on="user_id",how="left").fill_null(0)
    columns=cc_names+["weekly50","weekly80","weekly_decay"]
    prediction_days=(end-cutoff).total_seconds()/86400;history_days=(cutoff-TRAIN_START).total_seconds()/86400
    frame=augment_scores(frame,columns,prediction_days)
    features=base_features+context_new+columns+[prefix+c for prefix in ("log_","normalized_","rate_") for c in columns]
    # Keep counts usable for cold/partial-viewing distinctions, and expose real rates.
    extras=["meaningful_broadcasts","completed_broadcasts","observed_broadcasts"]
    frame=frame.with_columns([(pl.col(c)/history_days).log1p().alias("history_rate_"+c) for c in extras]+
        [(pl.col(c)/prediction_days).log1p().alias("prediction_rate_"+c) for c in ("score_exposure_10","score_exposure_80")]+
        [((pl.col("total_broadcasts_watched").exp()-1)/history_days).log1p().alias("programme_frequency_per_history_day")])
    features+=extras+["history_rate_"+c for c in extras]+["prediction_rate_score_exposure_10","prediction_rate_score_exposure_80","programme_frequency_per_history_day"]
    assert len(features)==len(set(features))
    frame=frame.with_columns([pl.col(c).cast(pl.Float32) for c in features]).sort(["user_id","tv_show_id"])
    assert np.isfinite(frame.select(features).to_numpy()).all()
    assert not frame.select("user_id","tv_show_id").is_duplicated().any()
    timing.update(total_feature_seconds=time.perf_counter()-started,rows=frame.height,new_candidates=missing.height)
    frame.write_parquet(path);save_json(info,{"features":features,"timing":timing,"cutoff":cutoff,"end":end})
    snap.update(frame=frame,features=features,timing=timing,pool=pool_from(frame))
    print(f"features {name}: {frame.height} candidates, +{missing.height}, {timing['total_feature_seconds']:.1f}s (weekly {timing['weekly_seconds']:.1f}s)",flush=True)


def fit(snaps):
    features=snaps[0]["features"];frames=[];labels=[]
    for s in snaps:
        assert s["features"]==features
        frames.append(s["frame"].select(["user_id","tv_show_id"]+features).with_columns((pl.lit(s["name"]+":")+pl.col("user_id")).alias("user_id")))
        labels.append(s["target"].select("user_id","tv_show_id","is_relevant").with_columns((pl.lit(s["name"]+":")+pl.col("user_id")).alias("user_id")))
    t=time.perf_counter();model=TemporalRanker().fit(pl.concat(frames),features,pl.concat(labels))
    return model,time.perf_counter()-t


def metrics(snap,recs):
    return dual_metrics(recs,snap["targets"],snap["target"],snap["submission"],snap["pool"])


def blend(frame,margins,anchor):
    p=margin_probabilities(frame,margins)
    f=frame.select("user_id").with_columns(pl.Series("a",anchor)).with_columns((pl.col("a")/pl.col("a").sum().over("user_id").clip(1e-12)).alias("a"))
    return .85*p+.15*f["a"].to_numpy()


def main():
    started=time.perf_counter()
    protected={str(p):sha256(p.read_bytes()).hexdigest() for root in ("outputs","evidence") for p in Path(root).glob("*")
        if p.is_file() and not p.name.startswith("phase10") and p.name not in ("submission_completion_v10.csv","submission_replay_v10.csv")}
    control=json.loads(Path("evidence/phase09_final_optimization.json").read_text())
    evidence={"created_msk":msk_now(),"methodology":{
        "review_verified":"Legacy clip_views and context_features allocate session seconds and broadcasts to start-hour slots; unchanged as historical control",
        "hourly_fix":"Union same-user/channel intervals, clip to cutoff, split UTC microsecond intervals on UTC hours, then label Europe/Kyiv weekday/hour including DST folds/gaps; broadcasts weighted by actual fractional airtime",
        "completion":"Observed broadcasts and >=80% / >=50% outcomes; channel prior shrunk toward user prior; finer context P=(completed+5*channel_prior)/(observed+5); sum future fractional airing weight*P/(past context opportunities+10)",
        "completion_contexts":["channel","channel x 3-hour","channel x duration bucket","channel x weekday x 3-hour"],
        "weekly":"Exact same-channel weekly-phase interval intersections; union overlaps including repeated DST hours, handle week wrap with offset-aligned historical weeks; mean historical-week >=50/80% completions and overlap with 90-day decay; sum actual known future airings",
        "efficiency":"Completion sparse prior-plus-residual matrices in 128-user blocks; weekly prefix-coverage integrals, no candidate-airing-history Cartesian join",
        "cohort":1260,"target":"Verified distinct >=80% airing frequency Top5; deterministic numeric-ID ties and empty users retained; official Kaggle definition remains unconfirmed",
        "anchor_weights_order":["exposure","conditional completion","weekly replay","item neighbors","personal frequency"],"anchor_grid":ANCHOR_WEIGHTS,
        "ranker":"Unchanged Phase8 LambdaRank defaults, one architecture; fixed B=.85*per-user margin softmax+.15*normalized new anchor",
        "selection":"Two disjoint windows only; a new completion model must not regress in either. Main overlapping window diagnostic only.",
        "optional_longer_snapshots":"Skipped to prioritize exact intervals, completion and replay deliverables","optional_binary":"Skipped; no specialist-pair retraining"},
        "results":{},"timing":{},"windows":{},"protected_sha256":protected}
    users=template_users();schedule=historical_schedule().collect();views=historical_viewing().collect()
    snaps=[]
    for name,cutoff,end in (("early",EARLY_CUTOFF,TRAIN_END),("short",TRAIN_END,date(6,22)),("late",date(6,22),VAL_END),("main",TRAIN_END,VAL_END)):
        s=snapshot(name,cutoff,end,schedule,views,users);s["submission"]=users;s["target"]=top_five_target(s["targets"])
        s.pop("frame",None);s.pop("profiles",None)
        s["schedule"]=schedule.filter((pl.col("programme_start_utc")<end)&(pl.col("programme_end_utc")>cutoff)&(pl.col("tv_show_id")>0)).unique(subset=["channel_id","programme_start_utc","tv_show_id"])
        build(s,views,schedule);snaps.append(s)
        evidence["timing"][name]=s["timing"];evidence["windows"][name]={"cutoff":cutoff,"label_end":end}
    early,short,late,main_snap=snaps;validations=[short,late,main_snap]
    for s in validations:
        evidence["results"][s["name"]]={"control_v5":control["results"][s["name"]]["V5_original"]["submission"]}
        for variant in ANCHOR_WEIGHTS:
            values=personal_anchor(s["frame"],variant)
            recs=ranked_predictions(s["frame"],values,s["users"],s["eligible"])
            evidence["results"][s["name"]]["anchor_"+variant]=metrics(s,recs)
    def value(window,name):return evidence["results"][window][name]["top5_frequency"]["map5"]
    anchor=max(ANCHOR_WEIGHTS,key=lambda n:np.mean([value(w,"anchor_"+n) for w in ("short","late")]))
    evidence["selected_anchor"]=anchor
    model_early,seconds=fit([early]);evidence["timing"]["fit_early_seconds"]=seconds
    model_late,seconds=fit([early,short]);evidence["timing"]["fit_late_seconds"]=seconds
    for s in validations:
        training=[early,short] if s["name"]=="late" else [early]
        assert max(t["end"] for t in training)<=s["cutoff"]
        model=model_late if s["name"]=="late" else model_early
        margins=model.model.predict(s["frame"].select(model.features).to_numpy().astype(np.float32))
        for name,values in (("A_completion",margins),("B_replay_blend",blend(s["frame"],margins,personal_anchor(s["frame"],anchor)))):
            evidence["results"][s["name"]][name]=metrics(s,ranked_predictions(s["frame"],values,s["users"],s["eligible"]))
        for name in ("control_v5","anchor_"+anchor,"A_completion","B_replay_blend"):
            print(f"{s['name']} {name}: {value(s['name'],name):.6f}",flush=True)
    valid=lambda name:all(value(w,name)>=value(w,"control_v5")-1e-12 for w in ("short","late"))
    candidates=[n for n in ("A_completion","B_replay_blend") if valid(n)]
    selected=max(candidates,key=lambda n:np.mean([value(w,n) for w in ("short","late")])) if candidates else "control_v5"
    replay="B_replay_blend" if valid("B_replay_blend") else ("anchor_"+anchor if valid("anchor_"+anchor) else selected)
    evidence["selection"]={"completion":selected,"replay":replay,"consistent_new_options":candidates,"main_not_used":True}
    save_json(OUT,evidence)
    del main_snap["frame"];gc.collect()
    production,seconds=fit([early,short,late]);production.model.booster_.save_model("outputs/phase10_completion_ranker.txt")
    evidence["timing"]["production_fit_seconds"]=seconds;evidence["model_parameters"]=production.model.get_params();evidence["features"]=production.features
    future=DataIngestion(PipelineConfig()).load_schedule("31-42").collect()
    final=snapshot("final",TEST_START,TEST_END,schedule,views,users,future=future)
    final.pop("frame",None);final.pop("profiles",None)
    final["schedule"]=future.filter((pl.col("programme_start_utc")<TEST_END)&(pl.col("programme_end_utc")>TEST_START)&(pl.col("tv_show_id")>0)).unique(subset=["channel_id","programme_start_utc","tv_show_id"])
    build(final,views,schedule);evidence["timing"]["final"]=final["timing"]
    f=final["frame"];margins=production.model.predict(f.select(production.features).to_numpy().astype(np.float32))
    av=personal_anchor(f,anchor);values={"A_completion":margins,"B_replay_blend":blend(f,margins,av),"anchor_"+anchor:av}
    import csv
    with open("outputs/submission_objective_v5.csv",newline="") as h:old={r["user_id"]:list(map(int,r["tv_show_id"].split())) for r in csv.DictReader(h)}
    checks={}
    for path,chosen in (("outputs/submission_completion_v10.csv",selected),("outputs/submission_replay_v10.csv",replay)):
        recs=old if chosen=="control_v5" else ranked_predictions(f,values[chosen],users,final["eligible"])
        write_submission(path,recs,final["eligible"]);checks[path]=validate_csv(path,final["eligible"])
    for path,digest in protected.items():assert sha256(Path(path).read_bytes()).hexdigest()==digest,path
    evidence.update({"completed_msk":msk_now(),"csv_checks":checks,"old_artifacts_unchanged":True,
        "production":{"history_end":TEST_START,"prediction_end":TEST_END,"label_snapshots":[[s["cutoff"],s["end"]] for s in (early,short,late)],"evaluated_on_refit_labels":False},
        "source_sha256":{p:sha256(Path(p).read_bytes()).hexdigest() for p in ("src/completion_replay.py","scripts/phase10_optimize.py")},
        "runtime_seconds":time.perf_counter()-started,"submitted_to_kaggle":False})
    save_json(OUT,evidence)
    print(f"Selected completion={selected}, replay={replay}; both CSVs validated; prior artifacts unchanged.",flush=True)


if __name__=="__main__":main()
