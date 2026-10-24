"""Phase-eight frequency targets, airing strength and sparse item retrieval."""
from pathlib import Path
from datetime import timedelta

import duckdb
import numpy as np
import polars as pl
from scipy.sparse import coo_matrix, csr_matrix, diags, vstack
from scipy.sparse.csgraph import connected_components

from .evaluation import average_precision_at_k, build_ground_truth
from .contextual_reranking import clip_views, _distribution


def merged_airing_segments(views, schedule, path="data/processed/phase08_segments.parquet"):
    """Union overlaps before clipping any snapshot; full airing denominators stay intact."""
    path = Path(path)
    if path.exists():
        return pl.scan_parquet(path)
    v = views.filter((pl.col("view_end_utc") > pl.col("view_start_utc")) & (pl.col("duraton_seconds") > 0)).unique()
    s = schedule.filter((pl.col("tv_show_id") > 0) & (pl.col("programme_duration_seconds") > 0)).unique(
        subset=["channel_id", "programme_start_utc", "tv_show_id"])
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    con.execute("SET threads=4")
    con.execute("SET memory_limit='5GB'")
    con.register("v", v.to_arrow())
    con.register("s", s.select("airing_id", "channel_id", "tv_show_id", "programme_start_utc", "programme_end_utc", "programme_duration_seconds").to_arrow())
    query = """
    WITH pieces AS (
      SELECT v.user_id,s.airing_id,s.channel_id,s.tv_show_id,s.programme_duration_seconds,
        greatest(v.view_start_utc,s.programme_start_utc) AS a,
        least(v.view_end_utc,s.programme_end_utc) AS b
      FROM v JOIN s ON v.channel_id=s.channel_id
        AND v.view_start_utc<s.programme_end_utc AND v.view_end_utc>s.programme_start_utc
    ), running AS (
      SELECT *,max(b) OVER (PARTITION BY user_id,airing_id ORDER BY a,b
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS previous_end FROM pieces
    ), groups AS (
      SELECT *,sum(CASE WHEN previous_end IS NULL OR a>previous_end THEN 1 ELSE 0 END)
        OVER (PARTITION BY user_id,airing_id ORDER BY a,b ROWS UNBOUNDED PRECEDING) AS g FROM running
    )
    SELECT user_id,airing_id,first(channel_id) AS channel_id,first(tv_show_id) AS tv_show_id,
      first(programme_duration_seconds) AS programme_duration_seconds,min(a) AS observed_start,max(b) AS observed_end
    FROM groups GROUP BY user_id,airing_id,g
    """
    try:
        con.execute(f"COPY ({query}) TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    finally:
        con.close()
    return pl.scan_parquet(path)


def airing_window(segments, start, end):
    f = segments.lazy() if isinstance(segments, pl.DataFrame) else segments
    return f.filter((pl.col("observed_start") < end) & (pl.col("observed_end") > start)).with_columns(
        pl.max_horizontal("observed_start", pl.lit(start)).alias("a"),
        pl.min_horizontal("observed_end", pl.lit(end)).alias("b"),
    ).with_columns((pl.col("b")-pl.col("a")).dt.total_seconds().alias("seconds")).group_by("user_id", "airing_id").agg(
        pl.col("channel_id").first(), pl.col("tv_show_id").first(), pl.col("programme_duration_seconds").first(),
        pl.col("seconds").sum(), pl.col("a").min().alias("first_observed"), pl.col("b").max().alias("last_observed"),
    ).with_columns(pl.min_horizontal("seconds", "programme_duration_seconds").alias("seconds")).with_columns(
        (pl.col("seconds")/pl.col("programme_duration_seconds")).alias("coverage")).collect()


def frequency_counts(airings):
    return airings.filter(pl.col("coverage") >= .8).group_by("user_id", "tv_show_id").agg(
        pl.col("airing_id").n_unique().alias("num_qualifying_broadcasts")).with_columns(pl.lit(True).alias("is_relevant"))


def top_five_target(counts):
    """Hypothesis, not a claim about the official Kaggle target. IDs break count ties."""
    return counts.filter((pl.col("num_qualifying_broadcasts") > 0) & (pl.col("tv_show_id") > 0)).sort(
        ["user_id", "num_qualifying_broadcasts", "tv_show_id"], descending=[False, True, False]).group_by(
            "user_id", maintain_order=True).head(5).with_columns(pl.lit(True).alias("is_relevant"))


def target_sizes(target, users):
    truth = build_ground_truth(target)
    sizes = np.array([len(truth.get(u, set())) for u in users])
    unique, counts = np.unique(sizes, return_counts=True)
    return {"users": len(users), "mean": float(sizes.mean()), "distribution": dict(zip(map(str, unique), map(int, counts)))}


def dual_metrics(recs, all_positive, top5, users, candidates=None):
    result = {}
    for name, target in (("all_positive", all_positive), ("top5_frequency", top5)):
        truth = build_ground_truth(target)
        ap, recall, ceiling, total, recovered = [], [], [], 0, 0
        for user in users:
            relevant = truth.get(user, set())
            ap.append(average_precision_at_k(recs[user], relevant))
            pool = set(candidates.get(user, [])) if candidates is not None else set(recs[user])
            hits = len(pool & relevant)
            total += len(relevant)
            recovered += hits
            recall.append(hits/len(relevant) if relevant else 0)
            ceiling.append(min(5, hits)/min(5, len(relevant)) if relevant else 0)
        result[name] = {"map5": float(np.mean(ap)), "candidate_recall_micro": recovered/max(total, 1),
                        "candidate_recall_macro_including_empty": float(np.mean(recall)),
                        "candidate_map5_ceiling": float(np.mean(ceiling)), "users": len(users)}
    return result


def strength_features(airings, cutoff):
    return airings.group_by("user_id", "tv_show_id").agg(
        (pl.col("coverage") >= .8).sum().alias("n80"),
        (pl.col("coverage") >= .5).sum().alias("n50"),
        (pl.col("coverage") >= .2).sum().alias("n20"),
        (pl.col("seconds") >= 300).sum().alias("n300"),
        ((pl.col("coverage") >= .5) & (pl.col("seconds") >= 60)).sum().alias("meaningful"),
        pl.col("seconds").sum().alias("strength_seconds"),
        ((pl.col("coverage") >= .8) & (pl.col("last_observed") > cutoff-timedelta(days=14))).sum().alias("qualified_14"),
        ((pl.col("coverage") >= .8) & (pl.col("last_observed") > cutoff-timedelta(days=28))).sum().alias("qualified_28"),
        ((pl.lit(cutoff)-pl.col("last_observed").max()).dt.total_seconds()/86400).alias("observed_days_ago"))


def item_neighbors(matrix, limit=80):
    """Cosine of binary meaningful co-viewing, blocked and popularity-normalized."""
    binary = matrix.copy()
    binary.data = np.ones_like(binary.data)
    frequency = np.asarray(binary.sum(axis=0)).ravel()
    norm = 1/np.sqrt(np.maximum(frequency, 1))
    normalized = binary @ diags(norm)
    blocks = []
    for lo in range(0, matrix.shape[1], 256):
        scores = (normalized[:, lo:lo+256].T @ normalized).toarray()
        rows, cols, values = [], [], []
        for row in range(len(scores)):
            scores[row, lo+row] = 0
            indices = np.flatnonzero(scores[row] > 0)
            indices = indices[np.lexsort((indices, -scores[row, indices]))][:limit]
            rows.extend([row]*len(indices)); cols.extend(indices); values.extend(scores[row, indices])
        blocks.append(coo_matrix((values, (rows, cols)), shape=scores.shape).tocsr())
    return vstack(blocks).tocsr() if blocks else csr_matrix((0, 0))


def normalized_rows(values):
    return values / np.maximum(values.max(axis=1, keepdims=True), 1e-12)


def stable_rank(values, ids, depth, positive_only=True):
    """Personal retrieval never proposes zero, negative or nonfinite evidence."""
    indices = np.flatnonzero(np.isfinite(values) & ((values > 0) if positive_only else True))
    return indices[np.lexsort((ids[indices], -values[indices]))][:depth]


def positive_candidate_union(scores, ids, global_order, pool_size=220):
    """Round-robin positive personal evidence first; complete popularity fallback last."""
    ordered = {name: stable_rank(values, ids, 100) for name, values in scores.items()}
    chosen, seen, owners = [], set(), {}
    for rank in range(max((len(order) for order in ordered.values()), default=0)):
        for name, order in ordered.items():
            if rank >= len(order):
                continue
            item = int(ids[order[rank]])
            if item not in seen:
                seen.add(item); chosen.append(item); owners[item] = name
                if len(chosen) == min(pool_size, len(ids)):
                    return chosen, owners, ordered
    for index in global_order:
        item = int(ids[index])
        if item not in seen:
            seen.add(item); chosen.append(item); owners[item] = "global_fallback"
            if len(chosen) == min(pool_size, len(ids)):
                break
    return chosen, owners, ordered


def retrieval(airings, strength, views, past, future, cutoff, users, eligible, old_ranks, pool_size=220):
    """Labels are deliberately absent. Future schedules supply only availability/exposure."""
    ids = np.array(sorted(eligible), dtype=np.int64)
    ui, ii = {u: j for j, u in enumerate(users)}, {int(p): j for j, p in enumerate(ids)}
    shape = (len(users), len(ids))
    scores = {}
    past_counts = past.group_by("tv_show_id", "channel_id").agg(pl.len().alias("past"))
    future_counts = future.group_by("tv_show_id", "channel_id").agg(pl.len().alias("future"))
    q = airings.filter(pl.col("coverage") >= .8).group_by("user_id", "tv_show_id", "channel_id").agg(
        pl.col("airing_id").n_unique().alias("qualifying"))
    q = q.join(past_counts, on=["tv_show_id", "channel_id"], how="left").join(
        future_counts, on=["tv_show_id", "channel_id"], how="inner").fill_null(0)
    for smooth in (10, 80):
        values = q.group_by("user_id", "tv_show_id").agg(
            (pl.col("qualifying")*pl.col("future")/(pl.col("past")+smooth)).sum().alias("score"))
        dense = np.zeros(shape, dtype=np.float32)
        for u, p, value in values.iter_rows():
            if u in ui and p in ii:
                dense[ui[u], ii[p]] = value
        scores[f"exposure_{smooth}"] = dense
    historical_ids = sorted(strength["tv_show_id"].unique().to_list())
    hi = {p: i for i, p in enumerate(historical_ids)}
    meaningful = strength.filter(pl.col("meaningful") > 0)
    graph_users = sorted(set(strength["user_id"]) | set(users))
    graph_ui = {u: i for i, u in enumerate(graph_users)}
    rows, cols, data = [], [], []
    for row in meaningful.iter_rows(named=True):
        rows.append(graph_ui[row["user_id"]]); cols.append(hi[row["tv_show_id"]])
        data.append(np.log1p(row["meaningful"])*2**(-row["observed_days_ago"]/60))
    matrix = coo_matrix((data, (rows, cols)), shape=(len(graph_users), len(historical_ids)), dtype=np.float32).tocsr()
    neighbors = item_neighbors(matrix)
    seeds = matrix[[graph_ui[u] for u in users]]
    dense = np.zeros(shape, dtype=np.float32)
    known = [(j, hi[int(p)]) for j, p in enumerate(ids) if int(p) in hi]
    if known:
        dest, source = zip(*known)
        for lo in range(0, len(users), 128):
            propagated = (seeds[lo:lo+128] @ neighbors[:, list(source)]).toarray()
            dense[lo:lo+128, list(dest)] = propagated
    scores["item_neighbors"] = dense
    recent_views = clip_views(views, cutoff-timedelta(days=28), cutoff)
    air = future.with_columns(
        pl.col("programme_start_utc").dt.convert_time_zone("Europe/Kyiv").dt.hour().cast(pl.Int64).alias("hour"),
        pl.col("programme_start_utc").dt.convert_time_zone("Europe/Kyiv").dt.weekday().cast(pl.Int64).alias("dow"),
        pl.lit(1.).alias("weight"))
    contextual = np.zeros(shape, dtype=np.float32)
    for keys, weight in ((["channel_id"], .5), (["channel_id", "hour"], .3), (["channel_id", "dow"], .2)):
        slots = {tuple(r): i for i, r in enumerate(pl.concat([recent_views.select(keys), air.select(keys)]).unique().sort(keys).iter_rows())}
        up = _distribution(recent_views, "user_id", keys, "seconds", users, slots)
        ip = _distribution(air, "tv_show_id", keys, "weight", ids.tolist(), slots)
        for lo in range(0, len(users), 128):
            contextual[lo:lo+128] += weight*(up[lo:lo+128] @ ip.T).toarray()
    scores["context_retrieval"] = contextual
    positive_pairs = {(u, int(p)) for u, p in strength.filter(pl.col("n80") > 0).select("user_id", "tv_show_id").iter_rows()}
    # Rank caches include fallback tails: PPR is genuinely positive only inside
    # the user's positive-interaction connected component, not for cold users.
    support = coo_matrix((np.ones(len(positive_pairs)),
        ([graph_ui[u] for u, p in positive_pairs], [len(graph_users)+hi[p] for u, p in positive_pairs])),
        shape=(len(graph_users)+len(historical_ids),)*2).tocsr()
    _, component = connected_components(support, directed=False)
    for name in ("ppr", "frequency", "recency"):
        dense = np.zeros(shape, dtype=np.float32)
        for u in users:
            for rank, p in enumerate(old_ranks[name].get(u, []), 1):
                personal = (u, p) in positive_pairs if name in ("frequency", "recency") else (
                    p in hi and component[graph_ui[u]] == component[len(graph_users)+hi[p]])
                if p in ii and personal:
                    dense[ui[u], ii[p]] = 1/(10+rank)
        scores[name] = dense
    popularity = strength.filter(pl.col("n80") > 0).group_by("tv_show_id").agg(pl.col("strength_seconds").sum().alias("popularity"))
    global_values = np.zeros(len(ids), dtype=np.float64)
    for p, value in popularity.iter_rows():
        if p in ii:
            global_values[ii[p]] = value
    global_order = stable_rank(global_values, ids, len(ids), positive_only=False)
    global_rank_score = np.empty(len(ids), dtype=np.float32)
    global_rank_score[global_order] = 1/(10+np.arange(1, len(ids)+1))
    scores["global"] = np.broadcast_to(global_rank_score, shape)
    recurrence = future.group_by("tv_show_id").len()
    vector = np.zeros(len(ids), dtype=np.float32)
    for p, count in recurrence.iter_rows():
        if p in ii:
            vector[ii[p]] = np.log1p(count)
    scores["recurrence"] = np.broadcast_to(vector, shape)
    # Preserve each positive personal source before filling from global popularity.
    source_names = ["frequency", "exposure_10", "exposure_80", "recency", "ppr", "item_neighbors", "context_retrieval"]
    contribution = {name: 0 for name in source_names+["global_fallback"]}
    offered = {name: 0 for name in source_names}
    fallback_users = 0
    pools, records = {}, []
    for row, user in enumerate(users):
        selected, owners, ordered = positive_candidate_union({n: scores[n][row] for n in source_names}, ids, global_order, pool_size)
        for owner in owners.values():
            contribution[owner] += 1
        for name, order in ordered.items():
            offered[name] += len(order)
        fallback_users += int("global_fallback" in owners.values())
        pools[user] = selected
        for p in selected:
            records.append((user, p, *[float(scores[n][row, ii[p]]) for n in scores]))
    columns = ["user_id", "tv_show_id"] + ["score_"+n for n in scores]
    frame = pl.DataFrame(records, schema=[(c, pl.String if c == "user_id" else pl.Int64 if c == "tv_show_id" else pl.Float64) for c in columns], orient="row")
    return pools, frame, {"neighbors_per_item_max": int(np.diff(neighbors.indptr).max(initial=0)),
                           "self_connections": int(np.count_nonzero(neighbors.diagonal())), "items": len(historical_ids),
                           "positive_only_personal_sources": True, "unique_first_owner_contributions": contribution,
                           "positive_proposals_top100": offered, "users_using_global_fill": fallback_users,
                           "pool_size": min(pool_size, len(ids))}


ANCHORS = {"personal": (.55, .20, .15, .05, .05), "balanced": (.40, .30, .15, .10, .05),
           "collaborative": (.25, .40, .20, .10, .05)}


def anchor_scores(frame, smoothing, variant):
    cols = [f"score_exposure_{smoothing}", "score_item_neighbors", "score_ppr", "score_context_retrieval", "score_recurrence"]
    result = np.zeros(frame.height)
    for col, weight in zip(cols, ANCHORS[variant]):
        # Log compression preserves meaningful strength while limiting dominant repeats.
        transformed = frame.select("user_id", pl.col(col).log1p().alias("v")).with_columns(
            (pl.col("v")/pl.col("v").max().over("user_id").clip(1e-12)).alias("normalized"))
        result += weight*transformed["normalized"].to_numpy()
    return result


def blend_scores(frame, margins, anchor, weight, transform):
    f = frame.select("user_id", "tv_show_id").with_columns(pl.Series("margin", margins), pl.Series("anchor", anchor))
    if transform == "rank":
        # Numeric-ID secondary ordering makes equal model scores reproducible.
        f = f.with_row_index("row").sort(["user_id", "margin", "tv_show_id"], descending=[False, True, False]).with_columns(
            (1/(10+pl.col("margin").cum_count().over("user_id"))).alias("m")).sort("row")
    elif transform == "exponential":
        f = f.with_columns(((pl.col("margin")-pl.col("margin").mean().over("user_id"))/
                           (pl.col("margin").std(ddof=0).over("user_id")+1e-6)).clip(-20, 20).exp().alias("m"))
    else:
        raise ValueError(transform)
    f = f.with_columns((pl.col("m")/pl.col("m").sum().over("user_id").clip(1e-12)).alias("m"),
                       (pl.col("anchor")/pl.col("anchor").sum().over("user_id").clip(1e-12)).alias("a"))
    return ((1-weight)*f["m"]+weight*f["a"]).to_numpy()
