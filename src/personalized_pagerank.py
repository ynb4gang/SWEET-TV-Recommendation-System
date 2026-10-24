"""Weighted bipartite Personalized PageRank, blocked sparse power iteration.

Restart is the user's frequency-weighted distribution on item nodes. The
two-step item iteration is algebraically the stationary bipartite walk:
q = (1-b)r + b P_iu.T P_ui.T q, b=(1-alpha)^2.
Full stationary masses are x_i=q/(2-alpha), x_u=(1-alpha)P_iu.T x_i.
Only bounded batches and top-200 item scores are retained on real datasets.
"""
import numpy as np
import polars as pl
from scipy.sparse import coo_matrix, csr_matrix, diags, bmat


class PersonalizedPageRankRecommender:
    def __init__(self, alpha=0.15, max_iter=200, tol=1e-6, edge_weight="frequency", batch_size=32):
        if not 0 < alpha < 1 or max_iter < 1 or tol <= 0:
            raise ValueError("Invalid PPR convergence parameters")
        if edge_weight not in ("frequency", "binary", "watch_time"):
            raise ValueError("Unknown edge weight")
        self.alpha, self.max_iter, self.tol = alpha, max_iter, tol
        self.edge_weight, self.batch_size = edge_weight, batch_size
        self._is_fitted = False

    def fit(self, train_interactions, eligible_programmes):
        frame = train_interactions.collect() if isinstance(train_interactions, pl.LazyFrame) else train_interactions
        if frame["user_id"].dtype != pl.String:
            raise ValueError("User IDs must be strings")
        self.eligible_programmes = set(eligible_programmes) - {0}
        positive = frame.filter(pl.col("is_relevant") & (pl.col("tv_show_id") > 0))
        self.user_ids = sorted(positive["user_id"].unique().to_list())
        self.programme_ids = sorted(positive["tv_show_id"].unique().to_list())
        self.user_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.programme_to_idx = {p: i for i, p in enumerate(self.programme_ids)}
        self.user_rankings, self.top_scores, self.ppr_scores = {}, {}, {}
        pop = positive.group_by("tv_show_id").agg(pl.col("user_id").n_unique().alias("n"))
        pop = pop.filter(pl.col("tv_show_id").is_in(sorted(self.eligible_programmes))).sort(
            ["n", "tv_show_id"], descending=[True, False])
        self.global_popularity_ranking = pop["tv_show_id"].to_list()
        self.global_popularity_ranking += sorted(self.eligible_programmes - set(self.global_popularity_ranking))
        nu, ni = len(self.user_ids), len(self.programme_ids)
        if not nu:
            self.transition_matrix = csr_matrix((0, 0))
            self.convergence = {"max_iterations": 0, "max_residual": 0, "unconverged_users": 0}
            self._is_fitted = True
            return self
        rows = np.fromiter((self.user_to_idx[u] for u in positive["user_id"]), dtype=np.int32)
        cols = np.fromiter((self.programme_to_idx[p] for p in positive["tv_show_id"]), dtype=np.int32)
        weight_col = "total_broadcasts_watched" if self.edge_weight == "frequency" else "total_watched_seconds"
        weights = np.ones(len(rows)) if self.edge_weight == "binary" else positive[weight_col].to_numpy().astype(float)
        if np.any(weights <= 0) or not np.isfinite(weights).all():
            raise ValueError("PPR weights must be finite and positive")
        w = coo_matrix((weights, (rows, cols)), shape=(nu, ni)).tocsr()
        self.ui = (diags(1 / np.asarray(w.sum(axis=1)).ravel()) @ w).tocsr()
        wt = w.T.tocsr()
        self.iu = (diags(1 / np.asarray(wt.sum(axis=1)).ravel()) @ wt).tocsr()
        self.transition_matrix = bmat([[None, self.ui], [self.iu, None]], format="csr")
        ids = np.array(self.programme_ids)
        mask = np.isin(ids, sorted(self.eligible_programmes))
        beta = (1-self.alpha)**2
        max_iterations, residual, unconverged = 0, 0.0, 0
        for lo in range(0, nu, self.batch_size):
            hi = min(nu, lo+self.batch_size)
            restart = self.ui[lo:hi].T.toarray()
            q = restart.copy()
            for iteration in range(1, self.max_iter+1):
                qnew = (1-beta)*restart + beta*(self.ui.T @ (self.iu.T @ q))
                delta = np.abs(qnew-q).sum(axis=0)
                q = qnew
                if delta.max() <= self.tol * (1-beta):
                    break
            unconverged += int((delta > self.tol * (1-beta)).sum())
            max_iterations = max(max_iterations, iteration)
            residual = max(residual, float(delta.max()))
            for j, u in enumerate(self.user_ids[lo:hi]):
                scores = q[:, j] / (2-self.alpha)
                indices = np.flatnonzero(mask & (scores > 0))
                indices = indices[np.lexsort((ids[indices], -scores[indices]))][:200]
                self.user_rankings[u] = ids[indices].tolist()
                self.top_scores[u] = {int(ids[i]): float(scores[i]) for i in indices}
                if nu <= 256:  # Full distributions only for small diagnostic fixtures.
                    self.ppr_scores[u] = np.concatenate(((1-self.alpha)*(self.iu.T @ scores), scores))
                else:
                    self.ppr_scores[u] = scores[indices].copy()
        self.convergence = {"max_iterations": max_iterations, "max_residual": residual,
                            "unconverged_users": unconverged}
        self._is_fitted = True
        return self

    def _compute_ppr(self, user_id):
        """Full distribution for independent stationary-equation checks."""
        restart = np.zeros(self.transition_matrix.shape[0])
        restart[len(self.user_ids):] = self.ui[self.user_to_idx[user_id]].toarray().ravel()
        x = restart.copy()
        for _ in range(self.max_iter*2):
            new = (1-self.alpha)*(self.transition_matrix.T @ x) + self.alpha*restart
            if np.abs(new-x).sum() < self.tol:
                return new
            x = new
        raise RuntimeError("PPR diagnostic did not converge")

    def recommend(self, user_id, k=5):
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted")
        ranked = self.user_rankings.get(user_id, [])[:k]
        seen = set(ranked)
        return (ranked + [p for p in self.global_popularity_ranking if p not in seen])[:k]

    def recommend_batch(self, user_ids, k=5):
        return {u: self.recommend(u, k) for u in user_ids}
