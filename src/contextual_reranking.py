"""Cutoff-isolated behavioral features and historically learned score blending."""
from datetime import timedelta

import numpy as np
import polars as pl
from scipy.sparse import coo_matrix, diags
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor

from .evaluation import average_precision_at_k, build_ground_truth


def clip_views(views, start, end):
    """No session seconds or timestamps beyond the observation boundary."""
    return views.filter((pl.col("view_start_utc") < end) & (pl.col("view_end_utc") > start)).with_columns(
        pl.max_horizontal("view_start_utc", pl.lit(start)).alias("view_start_utc"),
        pl.min_horizontal("view_end_utc", pl.lit(end)).alias("view_end_utc"),
    ).with_columns(
        (pl.col("view_end_utc") - pl.col("view_start_utc")).dt.total_seconds().alias("seconds"),
        pl.col("view_start_utc").dt.convert_time_zone("Europe/Kyiv").dt.date().alias("day"),
        pl.col("view_start_utc").dt.convert_time_zone("Europe/Kyiv").dt.hour().cast(pl.Int64).alias("hour"),
        pl.col("view_start_utc").dt.convert_time_zone("Europe/Kyiv").dt.weekday().cast(pl.Int64).alias("dow"),
    )


def cosine_stability(recent, previous, key, weight):
    """Cosine of two historical preference distributions; zero for no evidence."""
    a = recent.group_by(["user_id", key]).agg(pl.col(weight).cast(pl.Float64).sum().alias("a"))
    b = previous.group_by(["user_id", key]).agg(pl.col(weight).cast(pl.Float64).sum().alias("b"))
    joined = a.join(b, on=["user_id", key], how="full", coalesce=True).fill_null(0)
    return joined.group_by("user_id").agg(
        ((pl.col("a") * pl.col("b")).sum() /
         ((pl.col("a").pow(2).sum() * pl.col("b").pow(2).sum()).sqrt() + 1e-12)).alias("stability"))


def exposure_features(frame, history, recent, views, schedule, future, start, cutoff, end):
    """A: rates, recent programme observations, opportunities and stability."""
    original = set(frame.columns)
    history_days = (cutoff - start).total_seconds() / 86400
    future_days = (end - cutoff).total_seconds() / 86400
    past = schedule.filter((pl.col("programme_start_utc") < cutoff) & (pl.col("programme_end_utc") > start))
    opportunities = past.group_by("tv_show_id").agg(pl.len().alias("past_airings"))
    future_counts = future.group_by("tv_show_id").agg(pl.len().alias("future_airings"))
    observed = clip_views(views, start, cutoff)
    activity = observed.group_by("user_id").agg(
        pl.col("day").n_unique().alias("active_days"), pl.col("seconds").sum().alias("view_seconds"),
        pl.len().alias("sessions"))
    pairs = history.select("user_id", "tv_show_id", pl.col("total_broadcasts_watched").alias("raw_frequency"),
                           pl.col("num_qualifying_broadcasts").alias("raw_qualifying"))
    frame = frame.join(pairs, on=["user_id", "tv_show_id"], how="left").join(
        opportunities, on="tv_show_id", how="left").join(future_counts, on="tv_show_id", how="left").join(
        activity, on="user_id", how="left").fill_null(0)
    frame = frame.with_columns(
        (pl.col("raw_frequency") / (pl.col("active_days") + 1)).alias("frequency_per_active_day"),
        (pl.col("raw_frequency") / (pl.col("past_airings") + 1)).alias("consumption_per_opportunity"),
        (pl.col("raw_qualifying") / (pl.col("past_airings") + 1)).alias("completion_per_opportunity"),
        (pl.col("past_airings") / history_days).alias("historical_airings_per_day"),
        (pl.col("future_airings") / future_days).alias("future_airings_per_day"),
        (pl.col("active_days") / history_days).alias("active_day_fraction"),
        (pl.col("sessions") / history_days).log1p().alias("sessions_per_day"),
        (pl.col("view_seconds") / history_days / 3600).log1p().alias("hours_per_day"),
    ).with_columns(
        ((pl.col("future_airings_per_day") + .01) /
         (pl.col("historical_airings_per_day") + .01)).log().alias("airing_rate_shift"))
    user_profiles = activity.select("user_id", "active_days").with_columns(
        (pl.col("active_days") / history_days).alias("active_day_fraction"))
    for days in (7, 14, 28):
        length = min(days, history_days)
        rv = clip_views(views, max(start, cutoff - timedelta(days=days)), cutoff)
        act = rv.group_by("user_id").agg(pl.col("day").n_unique().alias(f"active_{days}"),
                                        pl.col("seconds").sum().alias(f"seconds_{days}"))
        r = recent[days].select("user_id", "tv_show_id",
                               pl.col("total_broadcasts_watched").alias(f"frequency_{days}"),
                               pl.col("num_qualifying_broadcasts").alias(f"qualifying_{days}"))
        opp = past.filter(pl.col("programme_end_utc") > cutoff - timedelta(days=days)).group_by("tv_show_id").agg(
            pl.len().alias(f"opportunities_{days}"))
        frame = frame.join(r, on=["user_id", "tv_show_id"], how="left").join(
            act, on="user_id", how="left").join(opp, on="tv_show_id", how="left").fill_null(0)
        frame = frame.with_columns(
            (pl.col(f"frequency_{days}") / length).alias(f"programme_rate_{days}"),
            (pl.col(f"frequency_{days}") / (pl.col(f"active_{days}") + 1)).alias(f"programme_active_rate_{days}"),
            (pl.col(f"qualifying_{days}") / (pl.col(f"opportunities_{days}") + 1)).alias(f"programme_exposure_rate_{days}"),
            (pl.col(f"active_{days}") / length).alias(f"active_fraction_{days}"),
            ((pl.col(f"seconds_{days}") / length + 60) /
             (pl.col("view_seconds") / history_days + 60)).log().alias(f"activity_shift_{days}"),
            ((pl.col(f"frequency_{days}") / length + .01) /
             (pl.col("raw_frequency") / history_days + .01)).log().alias(f"programme_shift_{days}"))
        if days == 14:
            user_profiles = user_profiles.join(act, on="user_id", how="left").join(
                activity.select("user_id", "view_seconds"), on="user_id", how="left").fill_null(0).with_columns(
                    ((pl.col("seconds_14") / length + 60) /
                     (pl.col("view_seconds") / history_days + 60)).log().alias("activity_shift"))
    # Programme stability uses disjoint 14-day observation windows, not labels.
    last = recent[14].select("user_id", "tv_show_id", pl.col("total_broadcasts_watched").alias("n"))
    previous = recent[28].select("user_id", "tv_show_id", pl.col("total_broadcasts_watched").alias("total")).join(
        last, on=["user_id", "tv_show_id"], how="left").fill_null(0).with_columns(
            (pl.col("total") - pl.col("n")).clip(0).alias("n"))
    # Airings crossing the boundary can appear in both; the difference is an explicit count proxy.
    stability = cosine_stability(last, previous, "tv_show_id", "n").rename({"stability": "programme_stability"})
    rv = clip_views(views, max(start, cutoff - timedelta(days=14)), cutoff)
    pv = clip_views(views, max(start, cutoff - timedelta(days=28)), cutoff - timedelta(days=14))
    cs = cosine_stability(rv, pv, "channel_id", "seconds").rename({"stability": "channel_stability"})
    user_profiles = user_profiles.join(stability, on="user_id", how="left").join(cs, on="user_id", how="left").fill_null(0)
    user_profiles = user_profiles.select("user_id", "active_day_fraction", "activity_shift", "programme_stability", "channel_stability")
    frame = frame.join(user_profiles.drop("active_day_fraction"), on="user_id", how="left").fill_null(0)
    excluded = {"raw_frequency", "raw_qualifying", "past_airings", "future_airings", "active_days", "view_seconds", "sessions"}
    excluded |= {f"{name}_{d}" for d in (7, 14, 28) for name in ("active", "seconds", "opportunities", "frequency", "qualifying")}
    features = [c for c in frame.columns if c not in original and c not in excluded]
    return frame.drop(list(excluded)), features, user_profiles


def _distribution(table, row_key, slot_keys, weight, row_ids, slots):
    grouped = table.group_by([row_key] + slot_keys).agg(pl.col(weight).sum().alias("weight"))
    ri = {v: i for i, v in enumerate(row_ids)}
    rows, cols, values = [], [], []
    for record in grouped.iter_rows(named=True):
        slot = tuple(record[k] for k in slot_keys)
        if record[row_key] in ri and slot in slots:
            rows.append(ri[record[row_key]])
            cols.append(slots[slot])
            values.append(record["weight"])
    mat = coo_matrix((values, (rows, cols)), shape=(len(row_ids), len(slots)), dtype=np.float32).tocsr()
    sums = np.asarray(mat.sum(axis=1)).ravel()
    return (diags(1 / np.maximum(sums, 1e-12)) @ mat).tocsr()


def context_features(frame, recent, views, schedule, future, start, cutoff):
    """B: all-airing-weighted channel/time compatibility, not dominant metadata."""
    users = sorted(frame["user_id"].unique().to_list())
    items = sorted(frame["tv_show_id"].unique().to_list())
    uid, iid = {u: i for i, u in enumerate(users)}, {p: i for i, p in enumerate(items)}
    ur = np.array([uid[u] for u in frame["user_id"]])
    ir = np.array([iid[p] for p in frame["tv_show_id"]])
    air = future.filter(pl.col("tv_show_id").is_in(items)).with_columns(
        pl.col("programme_start_utc").dt.convert_time_zone("Europe/Kyiv").dt.hour().cast(pl.Int64).alias("hour"),
        pl.col("programme_start_utc").dt.convert_time_zone("Europe/Kyiv").dt.weekday().cast(pl.Int64).alias("dow"),
        pl.lit(1.0).alias("weight"))
    specs = (("channel", ["channel_id"]), ("hour", ["hour"]), ("dow", ["dow"]),
             ("channel_hour", ["channel_id", "hour"]), ("channel_dow", ["channel_id", "dow"]))
    features = []
    for days in (7, 14, 28):
        observed = clip_views(views, max(start, cutoff - timedelta(days=days)), cutoff)
        for name, keys in specs:
            unique = pl.concat([observed.select(keys), air.select(keys)]).unique().sort(keys)
            slots = {tuple(v): i for i, v in enumerate(unique.iter_rows())}
            up = _distribution(observed, "user_id", keys, "seconds", users, slots)
            ip = _distribution(air, "tv_show_id", keys, "weight", items, slots)
            values = np.zeros(frame.height, dtype=np.float32)
            for lo in range(0, len(users), 128):
                hi = min(lo + 128, len(users))
                indices = np.flatnonzero((ur >= lo) & (ur < hi))
                compatibility = (up[lo:hi] @ ip.T).toarray()
                values[indices] = compatibility[ur[indices] - lo, ir[indices]]
            col = f"compatibility_{name}_{days}"
            frame = frame.with_columns(pl.Series(col, values))
            features.append(col)
        # Viewing-weighted historical broadcast length, not raw session length.
        past_duration = schedule.filter((pl.col("programme_start_utc") < cutoff) &
                                        (pl.col("programme_end_utc") > cutoff - timedelta(days=days))).group_by("tv_show_id").agg(
            pl.col("programme_duration_seconds").mean().alias("duration"))
        duration = recent[days].join(past_duration, on="tv_show_id", how="left").group_by("user_id").agg(
            ((pl.col("duration") * pl.col("total_broadcasts_watched")).sum() /
             pl.col("total_broadcasts_watched").sum().clip(1)).alias("preferred_duration"))
        future_duration = air.group_by("tv_show_id").agg(pl.col("programme_duration_seconds").mean().alias("future_duration"))
        frame = frame.join(duration, on="user_id", how="left").join(future_duration, on="tv_show_id", how="left")
        col = f"duration_mismatch_{days}"
        frame = frame.with_columns(((pl.col("future_duration") + 60).log() -
                                    (pl.col("preferred_duration") + 60).log()).abs().fill_null(0).alias(col)).drop(
                                        "preferred_duration", "future_duration")
        features.append(col)
    return frame, features


def ranked_predictions(frame, scores, users, eligible):
    ordered = frame.select("user_id", "tv_show_id").with_columns(pl.Series("score", scores)).sort(
        ["user_id", "score", "tv_show_id"], descending=[False, True, False])
    result = {u[0]: g["tv_show_id"].head(5).to_list() for u, g in ordered.partition_by("user_id", as_dict=True).items()}
    return {u: result.get(u, sorted(eligible)[:5]) for u in users}


class CalibratedAdaptiveBlend:
    """Monotone per-user margin calibration + shallow historical activity gate."""
    weights = (0.0, 0.02, 0.05, 0.10)
    profile_columns = ["active_day_fraction", "activity_shift", "programme_stability", "channel_stability"]

    @staticmethod
    def standardized(frame, scores):
        f = frame.select("user_id").with_columns(pl.Series("score", scores)).with_columns(
            ((pl.col("score") - pl.col("score").mean().over("user_id")) /
             (pl.col("score").std(ddof=0).over("user_id") + 1e-6)).alias("z"))
        return f["z"].to_numpy()

    def components(self, frame, scores):
        z = self.standardized(frame, scores)
        probability = expit(self.slope * z + self.intercept)
        prior = .5 * np.exp(-frame["rank_recency"].to_numpy() / 5) + .5 * np.exp(-frame["rank_frequency"].to_numpy() / 5)
        f = frame.select("user_id").with_columns(pl.Series("base", probability), pl.Series("prior", prior))
        f = f.with_columns((pl.col("base") / pl.col("base").sum().over("user_id").clip(1e-12)).alias("base"),
                           (pl.col("prior") / pl.col("prior").sum().over("user_id").clip(1e-12)).alias("prior"))
        return f["base"].to_numpy(), f["prior"].to_numpy()

    def fit(self, frame, scores, targets, profiles, users, eligible):
        truth = build_ground_truth(targets)
        positive = targets.filter(pl.col("is_relevant")).select("user_id", "tv_show_id").unique().with_columns(pl.lit(1).alias("label"))
        y = frame.select("user_id", "tv_show_id").join(positive, on=["user_id", "tv_show_id"], how="left", maintain_order="left").fill_null(0)["label"].to_numpy()
        z = self.standardized(frame, scores)
        # Equal total weight per query prevents prolific candidate users dominating calibration.
        count = frame.select(pl.len().over("user_id").alias("n"))["n"].to_numpy()
        calibrator = LogisticRegression(C=1, max_iter=100, random_state=42).fit(z.reshape(-1, 1), y, sample_weight=len(y) / len(users) / count)
        self.slope = max(float(calibrator.coef_[0, 0]), 1e-6)
        self.intercept = float(calibrator.intercept_[0])
        base, prior = self.components(frame, scores)
        aps = []
        for weight in self.weights:
            recs = ranked_predictions(frame, (1-weight)*base + weight*prior, users, eligible)
            aps.append([average_precision_at_k(recs[u], truth.get(u, set())) for u in users])
        values = np.asarray(aps).T
        self.training_mean_ap = values.mean(axis=0).tolist()
        x = pl.DataFrame({"user_id": users}).join(profiles, on="user_id", how="left", maintain_order="left").fill_null(0).select(self.profile_columns).to_numpy()
        self.gate = DecisionTreeRegressor(max_depth=2, min_samples_leaf=300, random_state=42).fit(x, values - values[:, :1])
        return self

    def recommend(self, frame, scores, profiles, users, eligible, fixed_weight=None):
        base, prior = self.components(frame, scores)
        if fixed_weight is None:
            x = pl.DataFrame({"user_id": users}).join(profiles, on="user_id", how="left", maintain_order="left").fill_null(0).select(self.profile_columns).to_numpy()
            choice = self.gate.predict(x).argmax(axis=1)
            weights = np.asarray(self.weights)[choice]
        else:
            weights = np.full(len(users), fixed_weight)
        self.usage = {str(w): int((weights == w).sum()) for w in self.weights}
        lookup = dict(zip(users, weights))
        row_weight = np.array([lookup[u] for u in frame["user_id"]])
        return ranked_predictions(frame, (1-row_weight)*base + row_weight*prior, users, eligible)
