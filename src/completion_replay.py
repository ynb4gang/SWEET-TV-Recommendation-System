"""Exact UTC interval allocation, local-week behavior and completion forecasting."""
from datetime import timedelta, date
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
from scipy.sparse import coo_matrix

from .constants import TRAIN_START
from .contextual_reranking import _distribution


def union_intervals(frame, keys, start, end, extra_end=None):
    """Union overlapping/touching intervals within each key, never across channels."""
    f = frame.sort(keys+[start, end]).with_columns(pl.col(end).cum_max().over(keys).alias("_running"))
    f = f.with_columns(pl.col("_running").shift(1).over(keys).alias("_previous"))
    f = f.with_columns((pl.col(start)>pl.col("_previous")).fill_null(True).cast(pl.Int64).alias("_new")).with_columns(
        pl.col("_new").cum_sum().over(keys).alias("_group"))
    agg = [pl.col(start).min(), pl.col(end).max()]
    if extra_end:
        agg.append(pl.col(extra_end).max())
    return f.group_by(keys+["_group"]).agg(agg).drop("_group")


def viewing_union(views, start, end):
    f = views.filter((pl.col("view_start_utc")<end)&(pl.col("view_end_utc")>start)&
        (pl.col("view_end_utc")>pl.col("view_start_utc"))).select(
        "user_id", "channel_id", pl.max_horizontal("view_start_utc",pl.lit(start)).alias("a"),
        pl.min_horizontal("view_end_utc",pl.lit(end)).alias("b"))
    return union_intervals(f,["user_id","channel_id"],"a","b")


def hourly_parts(frame, start="a", end="b"):
    """UTC arithmetic handles both DST folds and gaps; local labels follow splitting."""
    hour_us=3_600_000_000
    f = frame.with_columns(pl.col(start).dt.epoch("us").alias("_s"),pl.col(end).dt.epoch("us").alias("_e"))
    f = f.filter(pl.col("_e")>pl.col("_s")).with_columns(
        pl.int_ranges((pl.col("_s")//hour_us)*hour_us,((pl.col("_e")+hour_us-1)//hour_us)*hour_us,step=hour_us).alias("_hours")).explode("_hours")
    f = f.with_columns(pl.max_horizontal("_s","_hours").alias("_a"),pl.min_horizontal("_e",pl.col("_hours")+hour_us).alias("_b"))
    f = f.with_columns(pl.from_epoch("_a",time_unit="us").dt.replace_time_zone("UTC").alias("part_start"),
                      pl.from_epoch("_b",time_unit="us").dt.replace_time_zone("UTC").alias("part_end"),
                      ((pl.col("_b")-pl.col("_a"))/1_000_000).alias("seconds"))
    f = f.with_columns(pl.col("part_start").dt.convert_time_zone("Europe/Kyiv").alias("_local"))
    f = f.with_columns(pl.col("_local").dt.hour().cast(pl.Int64).alias("hour"),
        pl.col("_local").dt.weekday().cast(pl.Int64).alias("dow"),
        ((pl.col("_local").dt.date().cast(pl.Int32)-4)//7).alias("week_index"))
    f = f.with_columns(((pl.col("dow")-1)*86400+pl.col("hour")*3600+
        pl.col("_local").dt.minute().cast(pl.Int64)*60+pl.col("_local").dt.second().cast(pl.Int64)+
        pl.col("_local").dt.microsecond()/1_000_000).alias("phase_a"))
    return f.with_columns((pl.col("phase_a")+pl.col("seconds")).alias("phase_b"),(pl.col("hour")//3).alias("bucket3")).drop(
        "_s","_e","_hours","_a","_b","_local")


def scheduled_parts(schedule,start,end):
    f = schedule.filter((pl.col("programme_start_utc")<end)&(pl.col("programme_end_utc")>start)&
                        (pl.col("tv_show_id")>0)&(pl.col("programme_duration_seconds")>0)).unique(subset=["airing_id"])
    f = f.select("airing_id","channel_id","tv_show_id","programme_duration_seconds","programme_start_utc",
        pl.max_horizontal("programme_start_utc",pl.lit(start)).alias("a"),
        pl.min_horizontal("programme_end_utc",pl.lit(end)).alias("b"))
    f = hourly_parts(f).with_columns((pl.col("seconds")/pl.col("programme_duration_seconds")).alias("weight"),
        pl.when(pl.col("programme_duration_seconds")<=900).then(0).when(pl.col("programme_duration_seconds")<=1800).then(1)
        .when(pl.col("programme_duration_seconds")<=3600).then(2).when(pl.col("programme_duration_seconds")<=7200).then(3).otherwise(4).alias("duration_bucket"),
        ((pl.col("programme_start_utc").dt.convert_time_zone("Europe/Kyiv").dt.date().cast(pl.Int32)-4)//7).alias("_start_week"))
    return f.with_columns((pl.col("week_index")-pl.col("_start_week")).alias("week_offset")).with_columns(
        pl.col("week_offset").max().over("airing_id").alias("week_span"))


def gather_product(user_matrix,item_matrix,ur,ir):
    result=np.zeros(len(ur),dtype=np.float32)
    for lo in range(0,user_matrix.shape[0],128):
        rows=np.flatnonzero((ur>=lo)&(ur<lo+128))
        dense=(user_matrix[lo:lo+128]@item_matrix.T).toarray()
        result[rows]=dense[ur[rows]-lo,ir[rows]]
    return result


def exact_context(frame,union,future,cutoff):
    users=sorted(frame["user_id"].unique().to_list());items=sorted(frame["tv_show_id"].unique().to_list())
    ui={u:i for i,u in enumerate(users)};ii={p:i for i,p in enumerate(items)}
    ur=np.array([ui[u] for u in frame["user_id"]]);ir=np.array([ii[p] for p in frame["tv_show_id"]])
    new=[]
    for days in (7,14,28):
        recent=union.filter(pl.col("b")>cutoff-timedelta(days=days)).with_columns(
            pl.max_horizontal("a",pl.lit(cutoff-timedelta(days=days))).alias("a"))
        observed=hourly_parts(recent)
        for name,keys in (("channel",["channel_id"]),("hour",["hour"]),("dow",["dow"]),
            ("channel_hour",["channel_id","hour"]),("channel_dow",["channel_id","dow"]),
            ("channel_dow_hour",["channel_id","dow","hour"])):
            slots={tuple(r):i for i,r in enumerate(pl.concat([observed.select(keys),future.select(keys)]).unique().sort(keys).iter_rows())}
            up=_distribution(observed,"user_id",keys,"seconds",users,slots)
            ip=_distribution(future,"tv_show_id",keys,"weight",items,slots)
            col=f"compatibility_{name}_{days}"
            frame=frame.with_columns(pl.Series(col,gather_product(up,ip,ur,ir)))
            if name=="channel_dow_hour":new.append(col)
    return frame,new


def sparse_table(table,row_key,slot_keys,value,row_ids,slots):
    lookup={v:i for i,v in enumerate(row_ids)}
    rows=[];cols=[];values=[]
    for r in table.iter_rows(named=True):
        slot=tuple(r[k] for k in slot_keys)
        if r[row_key] in lookup and slot in slots:
            rows.append(lookup[r[row_key]]);cols.append(slots[slot]);values.append(r[value])
    return coo_matrix((values,(rows,cols)),shape=(len(row_ids),len(slots)),dtype=np.float32).tocsr()


def conditional_completion(base,airings,past,future,users,eligible,extra=20):
    """Sparse channel-prior decomposition avoids users x all behavioral slots joins."""
    items=sorted(eligible);ii={p:i for i,p in enumerate(items)}
    priors=airings.group_by("user_id","channel_id").agg(pl.len().alias("observed"),
        (pl.col("coverage")>=.8).sum().alias("n80"),(pl.col("coverage")>=.5).sum().alias("n50"))
    totals=priors.group_by("user_id").agg(pl.col("observed").sum().alias("total"),pl.col("n80").sum().alias("total80"),pl.col("n50").sum().alias("total50"))
    priors=priors.join(totals,on="user_id").with_columns(
        ((pl.col("n80")+2*(pl.col("total80")+1)/(pl.col("total")+2))/(pl.col("observed")+2)).alias("prior80"),
        ((pl.col("n50")+2*(pl.col("total50")+1)/(pl.col("total")+2))/(pl.col("observed")+2)).alias("prior50"))
    observed=airings.select("user_id","airing_id",(pl.col("coverage")>=.8).cast(pl.Float64).alias("q80"),
        (pl.col("coverage")>=.5).cast(pl.Float64).alias("q50")).join(
        past.select("airing_id","channel_id","bucket3","duration_bucket","dow","weight"),on="airing_id",how="inner")
    specs=(("channel",["channel_id"]),("time3",["channel_id","bucket3"]),
        ("duration",["channel_id","duration_bucket"]),("weektime",["channel_id","dow","bucket3"]))
    channels=sorted(set(past["channel_id"])|set(future["channel_id"]))
    cs={(c,):i for i,c in enumerate(channels)}
    models=[];names=[]
    for name,keys in specs:
        opp=past.group_by(keys).agg(pl.col("weight").sum().alias("opportunities"))
        forecasts=future.group_by(["tv_show_id"]+keys).agg(pl.col("weight").sum().alias("future_weight")).join(opp,on=keys,how="left").fill_null(0).with_columns(
            (pl.col("future_weight")/(pl.col("opportunities")+10)).alias("weight"))
        slots={tuple(r):i for i,r in enumerate(pl.concat([past.select(keys),future.select(keys)]).unique().sort(keys).iter_rows())}
        im=sparse_table(forecasts,"tv_show_id",keys,"weight",items,slots)
        ic=sparse_table(forecasts.group_by("tv_show_id","channel_id").agg(pl.col("weight").sum()),"tv_show_id",["channel_id"],"weight",items,cs)
        stats=observed.group_by(["user_id"]+keys).agg(pl.col("weight").sum().alias("n"),
            (pl.col("weight")*pl.col("q80")).sum().alias("c80"),(pl.col("weight")*pl.col("q50")).sum().alias("c50")).join(
                priors.select("user_id","channel_id","prior80","prior50"),on=["user_id","channel_id"])
        for q in (80,50):
            residual=stats.with_columns(((pl.col(f"c{q}")+5*pl.col(f"prior{q}"))/(pl.col("n")+5)-pl.col(f"prior{q}")).alias("residual"))
            um=sparse_table(residual,"user_id",keys,"residual",users,slots)
            uc=sparse_table(priors,"user_id",["channel_id"],f"prior{q}",users,cs)
            models.append((um,im,uc,ic));names.append(f"completion{q}_{name}")
    base_pool={u[0]:set(g["tv_show_id"].to_list()) for u,g in base.partition_by("user_id",as_dict=True).items()}
    records=[]
    for lo in range(0,len(users),128):
        blocks=[(um[lo:lo+128]@im.T+uc[lo:lo+128]@ic.T).toarray().clip(0) for um,im,uc,ic in models]
        for j,u in enumerate(users[lo:lo+128]):
            strength=sum(blocks[k][j]/max(float(blocks[k][j].max()),1e-12) for k in (0,2,4,6))
            positive=np.flatnonzero(strength>0)
            order=positive[np.lexsort((np.array(items)[positive],-strength[positive]))][:extra]
            selected=base_pool.get(u,set())|{items[i] for i in order}
            for p in sorted(selected):records.append((u,p,*[float(b[j,ii[p]]) for b in blocks]))
    frame=pl.DataFrame(records,schema=[("user_id",pl.String),("tv_show_id",pl.Int64)]+[(n,pl.Float32) for n in names],orient="row")
    meaningful=airings.group_by("user_id").agg(
        ((pl.col("coverage")>=.5)&(pl.col("seconds")>=60)).sum().alias("meaningful_broadcasts"),
        (pl.col("coverage")>=.8).sum().alias("completed_broadcasts"),pl.len().alias("observed_broadcasts"))
    return frame,names,meaningful


def weekly_replay(union,future,pairs,start,cutoff,temp_path):
    """Match exact phase intervals; align wrapped Monday pieces to the prior Sunday."""
    path=Path(temp_path)
    if path.exists():return pl.read_parquet(path)
    h=hourly_parts(union).select("user_id","channel_id","week_index","hour","dow","phase_a","phase_b",pl.col("part_end").alias("last_seen"))
    h=union_intervals(h,["user_id","channel_id","week_index","hour","dow"],"phase_a","phase_b","last_seen")
    base_week=(start.astimezone(ZoneInfo('Europe/Kyiv')).date()-date(1970,1,5)).days//7
    weeks=(cutoff.astimezone(ZoneInfo('Europe/Kyiv')).date()-start.astimezone(ZoneInfo('Europe/Kyiv')).date()).days/7
    h=h.with_columns((pl.col("week_index")-base_week).alias("week"),
        (-(pl.lit(cutoff)-pl.col("last_seen")).dt.total_seconds()/86400*np.log(2)/90).exp().alias("decay"))
    profiles={}
    for key,g in h.partition_by("user_id","channel_id",as_dict=True).items():
        g=g.sort(["week","phase_a"])
        a=g["week"].to_numpy()*604800+g["phase_a"].to_numpy()
        b=g["week"].to_numpy()*604800+g["phase_b"].to_numpy();w=g["decay"].to_numpy()
        profiles[key]=(a,b,np.r_[0,np.cumsum(b-a)],w,np.r_[0,np.cumsum((b-a)*w)])
    bychannel={key[0]:g for key,g in future.partition_by("channel_id",as_dict=True).items()}
    candidates={key[0]:g["tv_show_id"].to_numpy() for key,g in pairs.partition_by("user_id",as_dict=True).items()}
    records=[]
    for (user,channel),profile in profiles.items():
        if user not in candidates or channel not in bychannel:continue
        f=bychannel[channel]
        mask=np.isin(f["tv_show_id"].to_numpy(),candidates[user])
        if not mask.any():continue
        f=f.filter(pl.Series(mask))
        _,first,inverse=np.unique(f["airing_id"].to_numpy(),return_index=True,return_inverse=True)
        a=f["phase_a"].to_numpy();b=f["phase_b"].to_numpy();offset=f["week_offset"].to_numpy()
        duration=f["programme_duration_seconds"].to_numpy()[first];span=f["week_span"].to_numpy()[first]
        items=f["tv_show_id"].to_numpy()[first];count=len(first)
        week=np.arange(int(np.ceil(weeks)))[:,None]
        shift=(week+offset[None,:])*604800
        seconds=covered(profile,b[None,:]+shift)-covered(profile,a[None,:]+shift)
        weighted=covered(profile,b[None,:]+shift,True)-covered(profile,a[None,:]+shift,True)
        codes=inverse[None,:]+week*count
        sec=np.bincount(codes.ravel(),weights=seconds.ravel(),minlength=week.size*count).reshape(week.size,count)
        dec=np.bincount(codes.ravel(),weights=weighted.ravel(),minlength=week.size*count).reshape(week.size,count)
        valid=week<weeks-span[None,:]
        q50=((sec>=.5*duration[None,:])&valid).sum(axis=0)
        q80=((sec>=.8*duration[None,:])&valid).sum(axis=0)
        decayed=(dec/duration[None,:]*valid).sum(axis=0)
        denominator=np.maximum(weeks-span,1)
        unique,codes=np.unique(items,return_inverse=True)
        scores=[np.bincount(codes,weights=v/denominator,minlength=len(unique)) for v in (q50,q80,decayed)]
        records.extend((user,int(p),float(scores[0][j]),float(scores[1][j]),float(scores[2][j])) for j,p in enumerate(unique))
    result=pl.DataFrame(records,schema=[("user_id",pl.String),("tv_show_id",pl.Int64),("weekly50",pl.Float64),
        ("weekly80",pl.Float64),("weekly_decay",pl.Float64)],orient="row").group_by("user_id","tv_show_id").agg(
            pl.col("weekly50").sum(),pl.col("weekly80").sum(),pl.col("weekly_decay").sum())
    result.write_parquet(path)
    return result


def covered(profile,t,weighted=False):
    """Exact integral of disjoint weekly intervals at arbitrary phase endpoints."""
    a,b,prefix,weights,weighted_prefix=profile
    full=np.searchsorted(b,t,side="right")
    index=np.minimum(full,len(a)-1)
    partial=np.where(full<len(a),np.clip(t-a[index],0,b[index]-a[index]),0)
    return (weighted_prefix[full]+partial*weights[index]) if weighted else prefix[full]+partial


ANCHOR_WEIGHTS={"completion":(.25,.40,.15,.15,.05),"replay":(.20,.20,.40,.15,.05),"balanced":(.30,.25,.25,.15,.05)}


def personal_anchor(frame,variant):
    columns=["score_exposure_10","completion_component","weekly_component","score_item_neighbors","score_frequency"]
    f=frame.with_columns(pl.mean_horizontal([pl.col(f"normalized_completion80_{c}") for c in ("channel","time3","duration","weektime")]).alias("completion_component"),
        (pl.col("normalized_weekly80")+.5*pl.col("normalized_weekly50")+pl.col("normalized_weekly_decay")).alias("weekly_component"))
    result=np.zeros(f.height)
    for c,w in zip(columns,ANCHOR_WEIGHTS[variant]):
        v=f.select("user_id",pl.col(c).log1p().alias("v")).with_columns((pl.col("v")/pl.col("v").max().over("user_id").clip(1e-12)).alias("p"))["p"].to_numpy()
        result+=w*v
    cold=f["meaningful_broadcasts"].to_numpy()==0
    result[cold]=f["score_global"].to_numpy()[cold]
    return result


def augment_scores(frame,columns,prediction_days):
    return frame.with_columns([pl.col(c).log1p().alias("log_"+c) for c in columns]+
        [(pl.col(c)/pl.col(c).max().over("user_id").clip(1e-12)).alias("normalized_"+c) for c in columns]+
        [(pl.col(c)/prediction_days).log1p().alias("rate_"+c) for c in columns])


def missing_retrieval_scores(pairs,strength,airings,past,future,ranks):
    """Existing retrieval definitions on newly added completion candidates only."""
    from .objective_alignment import item_neighbors
    from scipy.sparse.csgraph import connected_components
    users=sorted(strength["user_id"].unique().to_list());ui={u:i for i,u in enumerate(users)}
    ids=sorted(strength["tv_show_id"].unique().to_list());ii={p:i for i,p in enumerate(ids)}
    meaningful=strength.filter(pl.col("meaningful")>0)
    rows=[];cols=[];weights=[]
    for r in meaningful.iter_rows(named=True):
        rows.append(ui[r["user_id"]]);cols.append(ii[r["tv_show_id"]]);weights.append(np.log1p(r["meaningful"])*2**(-r["observed_days_ago"]/60))
    matrix=coo_matrix((weights,(rows,cols)),shape=(len(users),len(ids)),dtype=np.float32).tocsr()
    neighbor=item_neighbors(matrix)
    eligible=set(future["tv_show_id"])
    popularity=strength.filter(pl.col("n80")>0).group_by("tv_show_id").agg(pl.col("strength_seconds").sum().alias("v"))
    pop=dict(popularity.iter_rows());ordered=sorted(eligible,key=lambda p:(-pop.get(p,0),p));global_rank={p:r for r,p in enumerate(ordered,1)}
    freq={key:{u:{p:r for r,p in enumerate(v,1)} for u,v in ranks[key].items()} for key in ("ppr","frequency","recency")}
    positives={(u,p) for u,p in strength.filter(pl.col("n80")>0).select("user_id","tv_show_id").iter_rows()}
    support=coo_matrix((np.ones(len(positives)),([ui[u] for u,p in positives],[len(users)+ii[p] for u,p in positives])),
        shape=(len(users)+len(ids),)*2).tocsr()
    _,component=connected_components(support,directed=False)
    recurrent=dict(future.group_by("tv_show_id").len().iter_rows())
    out=[]
    for key,group in pairs.partition_by("user_id",as_dict=True).items():
        u=key[0]
        knn=(matrix[ui[u]]@neighbor).toarray().ravel() if u in ui else np.zeros(len(ids))
        for p in group["tv_show_id"]:
            v=[]
            for source in ("ppr","frequency","recency"):
                rank=freq[source].get(u,{}).get(p)
                supported=(u,p) in positives if source!="ppr" else (u in ui and p in ii and component[ui[u]]==component[len(users)+ii[p]])
                v.append(1/(10+rank) if rank is not None and supported else 0.)
            out.append((u,p,float(knn[ii[p]]) if p in ii else 0.,*v,1/(10+global_rank[p]),np.log1p(recurrent[p])))
    raw=pl.DataFrame(out,schema=[("user_id",pl.String),("tv_show_id",pl.Int64)]+[(n,pl.Float64) for n in (
        "score_item_neighbors","score_ppr","score_frequency","score_recency","score_global","score_recurrence")],orient="row")
    pc=past.group_by("tv_show_id","channel_id").len().rename({"len":"past"})
    fc=future.group_by("tv_show_id","channel_id").len().rename({"len":"future"})
    q=airings.filter(pl.col("coverage")>=.8).group_by("user_id","tv_show_id","channel_id").len().rename({"len":"q"}).join(
        pc,on=["tv_show_id","channel_id"],how="left").join(fc,on=["tv_show_id","channel_id"],how="inner").fill_null(0)
    scores=q.group_by("user_id","tv_show_id").agg(*[(pl.col("q")*pl.col("future")/(pl.col("past")+s)).sum().alias(f"score_exposure_{s}") for s in (10,80)])
    return raw.join(scores,on=["user_id","tv_show_id"],how="left").fill_null(0).with_columns(pl.lit(0.).alias("score_context_retrieval"))
