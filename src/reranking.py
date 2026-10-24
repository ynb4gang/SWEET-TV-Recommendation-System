"""Compact temporal LambdaRank and activity-conditioned rank fusion."""
import numpy as np
import polars as pl
from sklearn.tree import DecisionTreeRegressor

from .evaluation import average_precision_at_k, build_ground_truth
from .optimization import rrf


def activity_features(history, users):
    stats = history.group_by("user_id").agg(
        pl.len().alias("items"), pl.col("total_broadcasts_watched").sum().alias("broadcasts"),
        pl.col("num_qualifying_broadcasts").sum().alias("qualifying"),
        pl.col("total_watched_seconds").sum().alias("seconds"),
        pl.col("mean_coverage_all").mean().alias("completion"))
    frame = pl.DataFrame({"user_id": users}).join(stats, on="user_id", how="left").fill_null(0)
    values = frame.drop("user_id").to_numpy().astype(float)
    values[:, :4] = np.log1p(values[:, :4])
    return values


class AdaptiveHybrid:
    """Learn a shallow user-activity gate ONLY on an earlier temporal snapshot."""
    weights = [(1.0, 0.0), (0.8, 0.2), (0.6, 0.4), (0.4, 0.6), (0.0, 1.0)]

    def fit(self, history, targets, users, rankings, eligible):
        truth = build_ground_truth(targets)
        aps = []
        for weight in self.weights:
            recs = rrf([rankings["recency"], rankings["ppr"]], weight, users, eligible)
            aps.append([average_precision_at_k(recs[u], truth.get(u, set())) for u in users])
        # Learn differences relative to Recency, avoiding dominance by user activity.
        scores = np.asarray(aps).T
        self.gate = DecisionTreeRegressor(max_depth=2, min_samples_leaf=200, random_state=42)
        self.gate.fit(activity_features(history, users), scores - scores[:, :1])
        return self

    def recommend(self, history, rankings, users, eligible, k=5):
        choice = self.gate.predict(activity_features(history, users)).argmax(axis=1)
        output = {}
        self.usage = {}
        for index, weights in enumerate(self.weights):
            cohort = [u for u, c in zip(users, choice) if c == index]
            self.usage[str(weights)] = len(cohort)
            output.update(rrf([rankings["recency"], rankings["ppr"]], weights, cohort, eligible, k))
        return output


def candidate_features(history, rankings, users, eligible, cutoff, schedule, depth=50):
    """Generate candidates/features without ever receiving target viewing labels."""
    pairs = set()
    for name, per_user in rankings.items():
        for u in users:
            pairs.update((u, int(p)) for p in per_user.get(u, [])[:depth] if p in eligible)
    frame = pl.DataFrame(sorted(pairs), schema=[("user_id", pl.String), ("tv_show_id", pl.Int64)], orient="row")
    rank_columns = []
    for name, per_user in rankings.items():
        col = f"rank_{name}"
        rank_columns.append(col)
        rows = [(u, int(p), r) for u in users for r, p in enumerate(per_user.get(u, [])[:100], 1)]
        table = pl.DataFrame(rows, schema=[("user_id", pl.String), ("tv_show_id", pl.Int64), (col, pl.Int64)], orient="row")
        frame = frame.join(table, on=["user_id", "tv_show_id"], how="left").with_columns(pl.col(col).fill_null(201))
    cols = ["user_id", "tv_show_id", "total_broadcasts_watched", "num_qualifying_broadcasts",
            "total_watched_seconds", "mean_coverage_all", "max_coverage_all", "last_interaction_utc"]
    frame = frame.join(history.select(cols), on=["user_id", "tv_show_id"], how="left")
    global_stats = history.group_by("tv_show_id").agg(
        pl.col("is_relevant").sum().alias("global_users"),
        pl.col("total_broadcasts_watched").sum().alias("global_frequency"))
    frame = frame.join(global_stats, on="tv_show_id", how="left")
    activity = history.group_by("user_id").agg(
        pl.col("total_broadcasts_watched").sum().alias("user_frequency"),
        pl.col("num_qualifying_broadcasts").sum().alias("user_qualifying"))
    frame = frame.join(activity, on="user_id", how="left")
    available = schedule.filter(pl.col("tv_show_id").is_in(sorted(eligible)))
    metadata = available.group_by("tv_show_id").agg(
        pl.len().alias("airing_opportunities"),
        pl.col("programme_duration_seconds").mean().alias("mean_duration"))
    frame = frame.join(metadata, on="tv_show_id", how="left")
    frame = frame.with_columns(
        ((pl.lit(cutoff) - pl.col("last_interaction_utc")).dt.total_seconds()/86400)
        .clip(0, 10000).fill_null(10000).alias("days_ago"))
    frame = frame.fill_null(0)
    frame = frame.with_columns(
        (pl.col("total_broadcasts_watched") / (pl.col("user_frequency")+1)).alias("frequency_share"),
        (pl.col("total_broadcasts_watched") * (-np.log(2)*pl.col("days_ago")/30).exp()).alias("recency_frequency"))
    numeric = ["total_broadcasts_watched", "num_qualifying_broadcasts", "total_watched_seconds",
               "global_users", "global_frequency", "user_frequency", "user_qualifying",
               "airing_opportunities", "mean_duration", "recency_frequency"]
    frame = frame.with_columns([pl.col(c).log1p() for c in numeric])
    features = rank_columns + numeric + ["mean_coverage_all", "max_coverage_all", "days_ago", "frequency_share"]
    return frame.select(["user_id", "tv_show_id"]+features), features


class TemporalRanker:
    def __init__(self, parameters=None):
        self.parameters = dict(parameters or {})

    def fit(self, frame, features, targets):
        import lightgbm as lgb
        positives = targets.filter(pl.col("is_relevant")).select("user_id", "tv_show_id").unique().with_columns(pl.lit(1).alias("label"))
        data = frame.join(positives, on=["user_id", "tv_show_id"], how="left").with_columns(pl.col("label").fill_null(0))
        data = data.filter(pl.col("label").sum().over("user_id") > 0).sort(["user_id", "tv_show_id"])
        self.features = features
        self.training_rows = data.height
        self.training_users = data["user_id"].n_unique()
        parameters = dict(objective="lambdarank", n_estimators=100, learning_rate=0.05,
                          num_leaves=15, min_child_samples=50, reg_lambda=2.0,
                          random_state=42, n_jobs=4, verbosity=-1,
                          deterministic=True, force_col_wise=True, lambdarank_truncation_level=10)
        parameters.update(self.parameters)
        self.model = lgb.LGBMRanker(**parameters)
        group = data.group_by("user_id", maintain_order=True).len()["len"].to_list()
        self.model.fit(data.select(features).to_numpy().astype(np.float32), data["label"].to_numpy(), group=group)
        return self

    def recommend(self, frame, users, eligible):
        score = self.model.predict(frame.select(self.features).to_numpy().astype(np.float32))
        ordered = frame.select("user_id", "tv_show_id").with_columns(pl.Series("score", score)).sort(
            ["user_id", "score", "tv_show_id"], descending=[False, True, False])
        result = {u[0]: group["tv_show_id"].head(5).to_list()
                  for u, group in ordered.partition_by("user_id", as_dict=True).items()}
        return {u: result.get(u, sorted(eligible)[:5]) for u in users}


def fit_production_ranker(early_frame, early_labels, late_frame, late_labels, features):
    """Refit after selection, preserving labels and query groups per snapshot.

    Namespaced internal query IDs never go to CSV. They prevent an April
    positive for (u,i) from becoming a July positive for that same pair.
    """
    frames, labels = [], []
    for prefix, frame, truth in (("early:", early_frame, early_labels),
                                  ("late:", late_frame, late_labels)):
        frames.append(frame.with_columns((pl.lit(prefix)+pl.col("user_id")).alias("user_id")))
        labels.append(truth.with_columns((pl.lit(prefix)+pl.col("user_id")).alias("user_id"),
                                        pl.lit(True).alias("is_relevant")))
    return TemporalRanker().fit(pl.concat(frames), features, pl.concat(labels))
