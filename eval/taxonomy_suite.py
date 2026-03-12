"""
taxonomy_suite.py  —  P1.5 + P1.6
=====================================
P1.5: Family recoverability test
  k-means on tactic co-occurrence from Regime 1 data.
  Jaccard ≥ 0.70 for ≥ 5 families required before publishing the conjunction claim.

P1.6: Taxonomy model-selection
  k=5, k=7, k=9 comparison on three criteria before Tier A claim.

§ references: M2 v1.18 §M2.2.6, §M2.12.3, CMA v4.1 §P1.5–P1.6 roadmap
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import math
import random
import statistics
from collections import defaultdict, Counter

from telemetry import TickRecord, M2Family, BIOLOGICAL_COLLAPSE_ORDER


# ──────────────────────────────────────────────────────────────
# Tactic co-occurrence matrix
# ──────────────────────────────────────────────────────────────

def build_cooccurrence_matrix(
    records: List[TickRecord],
    window: int = 5,
) -> Tuple[List[str], List[List[float]]]:
    """
    Build tactic co-occurrence matrix from TickRecord stream.
    Two tactics co-occur if they appear within `window` ticks for the same agent.

    Returns: (tactic_labels, matrix) where matrix[i][j] = co-occurrence count
    """
    # Collect all tactic classes
    tactic_set = sorted(set(r.tactic_class for r in records))
    tactic_idx = {t: i for i, t in enumerate(tactic_set)}
    n = len(tactic_set)

    # Per-agent sequences
    by_agent: Dict[str, List[str]] = defaultdict(list)
    for r in sorted(records, key=lambda r: (r.agent_id, r.tick)):
        by_agent[r.agent_id].append(r.tactic_class)

    matrix = [[0.0] * n for _ in range(n)]
    for seq in by_agent.values():
        for i, t in enumerate(seq):
            window_tactics = seq[max(0, i - window): i + window + 1]
            for t2 in window_tactics:
                if t != t2:
                    matrix[tactic_idx[t]][tactic_idx[t2]] += 1.0

    # Normalise rows
    for i in range(n):
        row_sum = sum(matrix[i])
        if row_sum > 0:
            matrix[i] = [v / row_sum for v in matrix[i]]

    return tactic_set, matrix


# ──────────────────────────────────────────────────────────────
# Lightweight k-means for tactic co-occurrence
# ──────────────────────────────────────────────────────────────

def kmeans_cluster(
    tactic_labels: List[str],
    matrix: List[List[float]],
    k: int,
    seed: int = 42,
    max_iter: int = 100,
) -> Dict[int, Set[str]]:
    """
    k-means over rows of the tactic co-occurrence matrix.
    Returns: {cluster_id: {tactic_class, ...}}
    """
    rng = random.Random(seed)
    n = len(tactic_labels)
    if n == 0 or k >= n:
        return {i: {tactic_labels[i]} for i in range(min(k, n))}

    def dist(a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    # k-means++ initialisation
    centroids = [matrix[rng.randrange(n)]]
    for _ in range(k - 1):
        dists = [min(dist(row, c) for c in centroids) for row in matrix]
        total = sum(dists)
        if total == 0:
            centroids.append(matrix[rng.randrange(n)])
        else:
            r = rng.random() * total
            cumulative = 0.0
            for i, d in enumerate(dists):
                cumulative += d
                if cumulative >= r:
                    centroids.append(matrix[i])
                    break

    assignments = [0] * n
    for _ in range(max_iter):
        new_assignments = [min(range(k), key=lambda c: dist(matrix[i], centroids[c])) for i in range(n)]
        if new_assignments == assignments:
            break
        assignments = new_assignments
        for c in range(k):
            members = [matrix[i] for i, a in enumerate(assignments) if a == c]
            if members:
                centroids[c] = [sum(col) / len(members) for col in zip(*members)]

    clusters: Dict[int, Set[str]] = defaultdict(set)
    for i, c in enumerate(assignments):
        clusters[c].add(tactic_labels[i])

    return dict(clusters)


# ──────────────────────────────────────────────────────────────
# Jaccard similarity: discovered clusters vs authored families
# ──────────────────────────────────────────────────────────────

def family_tactic_sets(records: List[TickRecord]) -> Dict[str, Set[str]]:
    """Map authored M2Family → set of tactic_class values observed in that family."""
    result: Dict[str, Set[str]] = defaultdict(set)
    for r in records:
        result[r.active_policy_family.value].add(r.tactic_class)
    return dict(result)


def best_jaccard_match(
    discovered_clusters: Dict[int, Set[str]],
    authored_families: Dict[str, Set[str]],
) -> Dict[str, Tuple[int, float]]:
    """
    For each authored family, find the discovered cluster with highest Jaccard similarity.
    Returns: {family_name: (best_cluster_id, jaccard_score)}
    """
    results = {}
    for family, fam_tactics in authored_families.items():
        best_cluster, best_j = -1, 0.0
        for cluster_id, cluster_tactics in discovered_clusters.items():
            intersection = len(fam_tactics & cluster_tactics)
            union = len(fam_tactics | cluster_tactics)
            j = intersection / union if union > 0 else 0.0
            if j > best_j:
                best_j, best_cluster = j, cluster_id
        results[family] = (best_cluster, best_j)
    return results


# ──────────────────────────────────────────────────────────────
# P1.5 — Family Recoverability Test
# ──────────────────────────────────────────────────────────────

@dataclass
class RecoverabilityResult:
    k:                   int
    jaccard_by_family:   Dict[str, float]   # family → best Jaccard score
    mean_jaccard:        float
    families_above_0_70: int
    recoverable:         bool               # Jaccard ≥ 0.70 for ≥ 5 families
    notes:               str = ""

    def summary(self) -> str:
        lines = [f"\n── Family Recoverability Test (k={self.k}) ──────────────────"]
        for fam, j in sorted(self.jaccard_by_family.items(), key=lambda x: -x[1]):
            flag = "✓" if j >= 0.70 else "✗"
            lines.append(f"  {flag} {fam:<12} Jaccard = {j:.3f}")
        lines.append(f"\n  Families ≥ 0.70:  {self.families_above_0_70} / {len(self.jaccard_by_family)}")
        lines.append(f"  Mean Jaccard:     {self.mean_jaccard:.3f}")
        verdict = "RECOVERABLE — conjunction claim supported" if self.recoverable \
                  else "NOT RECOVERABLE — taxonomy may be over-specified. Run model-selection."
        lines.append(f"  Verdict: {verdict}")
        if self.notes:
            lines.append(f"  Note: {self.notes}")
        return "\n".join(lines)


def run_recoverability_test(
    records: List[TickRecord],
    k: int = 7,
    seed: int = 42,
) -> RecoverabilityResult:
    """
    P1.5: k-means + Jaccard comparison.
    Run on Regime 1 data. ~2h on full population run.

    If Jaccard < 0.70 for most families, run taxonomy model-selection (P1.6)
    before concluding that the 7-family spec is wrong.
    """
    if not records:
        return RecoverabilityResult(
            k=k, jaccard_by_family={}, mean_jaccard=0.0,
            families_above_0_70=0, recoverable=False,
            notes="No records provided."
        )

    labels, matrix = build_cooccurrence_matrix(records)
    if len(labels) < k:
        return RecoverabilityResult(
            k=k, jaccard_by_family={}, mean_jaccard=0.0,
            families_above_0_70=0, recoverable=False,
            notes=f"Only {len(labels)} tactic classes observed — need ≥ {k} for k-means."
        )

    clusters  = kmeans_cluster(labels, matrix, k=k, seed=seed)
    authored  = family_tactic_sets(records)
    matches   = best_jaccard_match(clusters, authored)

    jaccard_by_family = {f: j for f, (_, j) in matches.items()}
    above_threshold = sum(1 for j in jaccard_by_family.values() if j >= 0.70)
    mean_j = statistics.mean(jaccard_by_family.values()) if jaccard_by_family else 0.0
    recoverable = above_threshold >= 5

    result = RecoverabilityResult(
        k=k,
        jaccard_by_family=jaccard_by_family,
        mean_jaccard=mean_j,
        families_above_0_70=above_threshold,
        recoverable=recoverable,
    )
    print(result.summary())
    return result


# ──────────────────────────────────────────────────────────────
# P1.6 — Taxonomy Model-Selection
# ──────────────────────────────────────────────────────────────

def compute_bic(records: List[TickRecord], k: int, seed: int = 42) -> float:
    """
    Approximate BIC for k-family clustering on tactic co-occurrence data.
    Lower BIC = better fit per degree of freedom.
    BIC = -2 * log_likelihood + k * log(n)
    """
    if not records:
        return float("inf")
    labels, matrix = build_cooccurrence_matrix(records)
    if len(labels) < k:
        return float("inf")

    clusters = kmeans_cluster(labels, matrix, k=k, seed=seed)
    n = len(labels)
    # Inertia as proxy for -2 * log_likelihood
    centroids = {}
    for cluster_id, members in clusters.items():
        member_rows = [matrix[labels.index(t)] for t in members if t in labels]
        if member_rows:
            centroids[cluster_id] = [sum(col) / len(member_rows) for col in zip(*member_rows)]

    inertia = 0.0
    for i, label in enumerate(labels):
        row = matrix[i]
        # Find cluster
        for cluster_id, members in clusters.items():
            if label in members:
                centroid = centroids.get(cluster_id, row)
                inertia += sum((r - c) ** 2 for r, c in zip(row, centroid))
                break

    # BIC approximation
    log_lik = -inertia * n
    bic = -2 * log_lik + k * math.log(max(n, 1))
    return bic


def spearman_with_biological_prior(
    records: List[TickRecord],
    discovered_clusters: Dict[int, Set[str]],
) -> float:
    """
    Correlation between discovered cluster collapse order and biological prior.
    BIOLOGICAL_COLLAPSE_ORDER: REPAIR(0) < EXPLORE(1) < SEEK_HELP < DECEIVE < DOMINATE < WITHDRAW < DEFEND(6)
    """
    authored = family_tactic_sets(records)
    matches = best_jaccard_match(discovered_clusters, authored)

    # Build lists: authored family index in biological order vs cluster ID (proxy for discovered order)
    bio_rank = {f.value: i for i, f in enumerate(BIOLOGICAL_COLLAPSE_ORDER)}
    pairs = []
    for family, (cluster_id, j_score) in matches.items():
        if j_score > 0.30 and family in bio_rank:
            pairs.append((bio_rank[family], cluster_id))

    if len(pairs) < 4:
        return 0.0

    bio_vals  = [p[0] for p in pairs]
    disc_vals = [p[1] for p in pairs]

    def rank(lst):
        sorted_lst = sorted(enumerate(lst), key=lambda x: x[1])
        ranks = [0.0] * len(lst)
        for rank_val, (orig_idx, _) in enumerate(sorted_lst):
            ranks[orig_idx] = float(rank_val + 1)
        return ranks

    br = rank(bio_vals)
    dr = rank(disc_vals)
    n = len(br)
    mean_b = sum(br) / n
    mean_d = sum(dr) / n
    num = sum((br[i] - mean_b) * (dr[i] - mean_d) for i in range(n))
    den = math.sqrt(sum((br[i] - mean_b) ** 2 for i in range(n)) *
                    sum((dr[i] - mean_d) ** 2 for i in range(n)))
    return num / den if den else 0.0


@dataclass
class ModelSelectionResult:
    k:                   int
    bic:                 float
    spearman_bio:        float       # correlation with biological collapse prior
    families_recoverable: int        # Jaccard ≥ 0.70 family count
    notes:               str = ""


@dataclass
class TaxonomyModelSelectionResult:
    results:             List[ModelSelectionResult]
    recommended_k:       int
    recommendation_basis: str

    def summary(self) -> str:
        lines = ["\n── Taxonomy Model-Selection (P1.6) ──────────────────"]
        lines.append(f"  {'k':<4} {'BIC':>10} {'Spearman ρ':>12} {'Recoverable':>13}")
        lines.append("  " + "-" * 42)
        for r in self.results:
            flag = " ◀ recommended" if r.k == self.recommended_k else ""
            lines.append(
                f"  k={r.k:<3} BIC={r.bic:>10.1f}  ρ={r.spearman_bio:>8.3f}  "
                f"families={r.families_recoverable:>3}{flag}"
            )
        lines.append(f"\n  Recommended: k={self.recommended_k}  ({self.recommendation_basis})")
        return "\n".join(lines)


def run_taxonomy_model_selection(
    records: List[TickRecord],
    k_values: List[int] = None,
    seed: int = 42,
) -> TaxonomyModelSelectionResult:
    """
    P1.6: Compare k=5, k=7, k=9 on BIC, biological Spearman, and recoverability.
    The 7-family spec is a hypothesis, not a constraint — let the data decide.
    """
    if k_values is None:
        k_values = [5, 7, 9]

    results = []
    for k in k_values:
        bic = compute_bic(records, k, seed)

        labels, matrix = build_cooccurrence_matrix(records)
        clusters = kmeans_cluster(labels, matrix, k=k, seed=seed) if len(labels) >= k else {}
        spearman = spearman_with_biological_prior(records, clusters) if clusters else 0.0

        # Recoverability at this k
        rec = run_recoverability_test(records, k=k, seed=seed)
        results.append(ModelSelectionResult(
            k=k,
            bic=bic,
            spearman_bio=spearman,
            families_recoverable=rec.families_above_0_70,
        ))

    # Recommend: lowest BIC with ≥ 5 recoverable families and ρ > 0.60
    candidates = [r for r in results if r.families_recoverable >= 5 and r.spearman_bio > 0.60]
    if candidates:
        recommended = min(candidates, key=lambda r: r.bic)
        basis = "lowest BIC with ≥ 5 recoverable families and ρ > 0.60"
    else:
        # Fall back to highest Spearman
        recommended = max(results, key=lambda r: r.spearman_bio)
        basis = "highest biological Spearman ρ (no k met recoverability threshold)"

    result = TaxonomyModelSelectionResult(
        results=results,
        recommended_k=recommended.k,
        recommendation_basis=basis,
    )
    print(result.summary())
    return result


# ──────────────────────────────────────────────────────────────
# P1.7 — Parameter Perturbation Sweeps
# ──────────────────────────────────────────────────────────────

@dataclass
class PerturbationResult:
    parameter:          str
    perturbation_pct:   float       # ±0.05 = ±5%
    baseline_metric:    float
    perturbed_metric:   float
    relative_change:    float       # |perturbed - baseline| / baseline
    stable:             bool        # relative_change ≤ 0.30 (30% tolerance)


@dataclass
class PerturbationSuiteResult:
    results:            List[PerturbationResult]
    all_stable:         bool
    unstable_parameters: List[str]

    def summary(self) -> str:
        lines = ["\n── Perturbation Sweeps ±5% (P1.7) ──────────────────"]
        for r in self.results:
            flag = "✓" if r.stable else "✗ UNSTABLE"
            lines.append(
                f"  {flag:<12} {r.parameter:<35} "
                f"Δ={r.relative_change:+.1%} (target ≤ 30%)"
            )
        if self.unstable_parameters:
            lines.append(f"\n  ⚠ Unstable: {self.unstable_parameters}")
            lines.append("    Recalibrate before publishing. Do not treat as hyperparameter search.")
        else:
            lines.append("\n  All parameters stable ≤ 30% — falsifiability gate passed.")
        return "\n".join(lines)


def run_perturbation_sweeps(
    run_fn,        # callable(config_override: dict) → List[TickRecord]
    base_config: dict,
    parameters_to_sweep: Optional[List[str]] = None,
    perturbation: float = 0.05,
    metric_fn=None,
) -> PerturbationSuiteResult:
    """
    P1.7: ±5% perturbation sweep over primary A/Q parameters.
    For each parameter: run baseline, run +5%, run -5%, check metric stability.

    metric_fn: callable(records) → float
              defaults to tactic_class_cv
    """
    from baseline_suite import tactic_class_cv as default_metric
    if metric_fn is None:
        metric_fn = default_metric

    if parameters_to_sweep is None:
        parameters_to_sweep = [
            "families.REPAIR.rd_threshold",
            "families.EXPLORE.rd_threshold",
            "families.SEEK_HELP.rd_threshold",
            "families.DOMINATE.rd_threshold",
            "families.DEFEND.rd_threshold",
            "baseline_controller.activation_threshold",
            "persistence_minimum_ticks",
        ]

    baseline_records = run_fn(base_config)
    baseline_val = metric_fn(baseline_records)

    results: List[PerturbationResult] = []
    for param in parameters_to_sweep:
        for direction in [+perturbation, -perturbation]:
            # Build perturbed config
            perturbed = _deep_copy_config(base_config)
            _set_nested(perturbed, param, _get_nested(base_config, param) * (1.0 + direction))

            try:
                perturbed_records = run_fn(perturbed)
                perturbed_val = metric_fn(perturbed_records)
                rel_change = abs(perturbed_val - baseline_val) / max(1e-6, abs(baseline_val))
            except Exception as e:
                perturbed_val = float("nan")
                rel_change = float("inf")

            results.append(PerturbationResult(
                parameter=f"{param} {'+' if direction > 0 else ''}{direction:.0%}",
                perturbation_pct=direction,
                baseline_metric=baseline_val,
                perturbed_metric=perturbed_val,
                relative_change=rel_change,
                stable=rel_change <= 0.30,
            ))

    all_stable = all(r.stable for r in results)
    unstable = [r.parameter for r in results if not r.stable]
    result = PerturbationSuiteResult(results=results, all_stable=all_stable, unstable_parameters=unstable)
    print(result.summary())
    return result


# ── Config helpers ────────────────────────────────────────────

def _deep_copy_config(cfg: dict) -> dict:
    import json
    return json.loads(json.dumps(cfg))


def _get_nested(cfg: dict, path: str):
    parts = path.split(".")
    v = cfg
    for p in parts:
        v = v[p]
    return v


def _set_nested(cfg: dict, path: str, value) -> None:
    parts = path.split(".")
    v = cfg
    for p in parts[:-1]:
        v = v[p]
    v[parts[-1]] = value
