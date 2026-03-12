"""
topology_suite.py  —  P2.4 (Suite 1 of 3)
==========================================
Hostile-control topology evaluation: M2 vs LSM.
Tests whether typed policy families produce lower-entropy, seed-stable
collapse topology compared to the matched LSM baseline.

Four measurements:
  1. Sequence KL/JS divergence (distribution over collapse signatures)
  2. Spearman rank vs biological prior
  3. Seed instability (pairwise Kendall-like disagreement)
  4. Accessibility surface KL over rd bins

Plus: Hysteresis metric + state volatility (A.90).

Battery win condition (topology_win):
  M2 lower seed instability AND higher Spearman AND JS divergence > 0.10

numpy-only. No torch.
"""

from __future__ import annotations
import math
import random
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict

import numpy as np

from agent_wrapper import AgentWrapper

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

# Updated to match M2 v1.18 taxonomy (SUBMIT dropped, SEEK_HELP added)
M2_FAMILIES = ["DEFEND", "WITHDRAW", "REPAIR", "EXPLORE", "DOMINATE", "SEEK_HELP", "DECEIVE"]
NUM_STATES   = len(M2_FAMILIES)

# Biological collapse prior: earliest to collapse listed first
# REPAIR(2) → EXPLORE(3) → SEEK_HELP(5) → DECEIVE(6) → DOMINATE(4) → WITHDRAW(1) → DEFEND(0)
BIOLOGICAL_COLLAPSE_PRIOR: Tuple[int, ...] = (2, 3, 5, 6, 4, 1, 0)


# ──────────────────────────────────────────────────────────────
# Numpy math utilities
# ──────────────────────────────────────────────────────────────

def safe_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.maximum(x.astype(np.float64), 0.0)
    s = x.sum()
    return x / s if s > eps else np.ones_like(x) / max(len(x), 1)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p, q = safe_normalize(p), safe_normalize(q)
    m = 0.5 * (p + q)
    eps = 1e-12
    kl_pm = float(np.sum(p * np.log((p + eps) / (m + eps))))
    kl_qm = float(np.sum(q * np.log((q + eps) / (m + eps))))
    return 0.5 * (kl_pm + kl_qm)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    p = safe_normalize(p + eps)
    q = safe_normalize(q + eps)
    return float(np.sum(p * np.log(p / q)))


def pairwise_js_matrix(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """P: [K, A], Q: [K, A] → [K, K]"""
    K = P.shape[0]
    out = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            out[i, j] = js_divergence(P[i], Q[j])
    return out


def best_permutation_min_cost(cost: np.ndarray) -> List[int]:
    K = cost.shape[0]
    best_perm, best_score = None, float("inf")
    for perm in itertools.permutations(range(K)):
        score = sum(float(cost[i, perm[i]]) for i in range(K))
        if score < best_score:
            best_score, best_perm = score, list(perm)
    return best_perm


def spearman_corr(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    def rank(lst):
        s = sorted(range(n), key=lambda i: lst[i])
        r = [0.0] * n
        for rank_val, orig in enumerate(s):
            r[orig] = float(rank_val + 1)
        return r
    xr, yr = rank(xs), rank(ys)
    mx = sum(xr) / n
    my = sum(yr) / n
    num = sum((xr[i] - mx) * (yr[i] - my) for i in range(n))
    den = math.sqrt(sum((xr[i] - mx) ** 2 for i in range(n)) *
                    sum((yr[i] - my) ** 2 for i in range(n)))
    return num / den if den else 0.0


# ──────────────────────────────────────────────────────────────
# Stress schedule
# ──────────────────────────────────────────────────────────────

@dataclass
class CatastropheSchedule:
    warmup_ticks:   int   = 30
    ramp_ticks:     int   = 60
    peak_ticks:     int   = 80
    recovery_ticks: int   = 40
    rd_start:       float = 0.05
    rd_peak:        float = 0.95
    shock_noise_std: float = 0.01

    @property
    def total_ticks(self) -> int:
        return self.warmup_ticks + self.ramp_ticks + self.peak_ticks + self.recovery_ticks

    @property
    def recovery_phase_start(self) -> int:
        return self.warmup_ticks + self.ramp_ticks + self.peak_ticks

    def build(self, seed: int = 0) -> np.ndarray:
        rng  = np.random.default_rng(seed)
        warm = np.full(self.warmup_ticks, self.rd_start)
        ramp = np.linspace(self.rd_start, self.rd_peak, self.ramp_ticks)
        peak = np.full(self.peak_ticks, self.rd_peak)
        rec  = np.linspace(self.rd_peak, 0.45, self.recovery_ticks)
        rd   = np.concatenate([warm, ramp, peak, rec])
        if self.shock_noise_std > 0:
            rd = np.clip(rd + rng.normal(0, self.shock_noise_std, size=len(rd)), 0.0, 1.0)
        return rd.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Collapse trace
# ──────────────────────────────────────────────────────────────

@dataclass
class CollapseTrace:
    masks:           np.ndarray   # [T, K] bool accessible
    active:          np.ndarray   # [T] int state index
    rd:              np.ndarray   # [T] float
    collapse_order:  Tuple[int, ...]
    recovery_order:  Tuple[int, ...]
    collapse_sig:    str
    recovery_sig:    str
    hysteresis_inversions: int


def extract_collapse_trace(
    masks:   np.ndarray,   # [T, K]
    active:  np.ndarray,   # [T]
    rd:      np.ndarray,   # [T]
    schedule: Optional[CatastropheSchedule] = None,
) -> CollapseTrace:
    T, K = masks.shape

    # Collapse: first tick each state becomes inaccessible
    first_drop = np.full(K, T, dtype=np.int32)
    for k in range(K):
        drops = np.where(masks[:, k] == 0)[0]
        if len(drops) > 0:
            first_drop[k] = int(drops[0])

    persistence = masks.astype(float).mean(axis=0)
    sortable = sorted([(int(first_drop[k]), float(persistence[k]), k) for k in range(K)])
    collapse_order = tuple(k for _, _, k in sortable)

    # Recovery
    rec_start = schedule.recovery_phase_start if schedule else 0
    recovery_tick = np.full(K, T, dtype=np.int32)
    for k in range(K):
        window = masks[rec_start:, k]
        up = np.where(window == 1)[0]
        if len(up) > 0:
            recovery_tick[k] = int(up[0]) + rec_start

    rec_sortable = sorted([(int(recovery_tick[k]), k) for k in range(K)])
    recovery_order = tuple(k for _, k in rec_sortable)

    # Hysteresis inversions: recovery should be approximately inverse of collapse
    collapse_pos  = {k: i for i, k in enumerate(collapse_order)}
    recovery_pos  = {k: i for i, k in enumerate(recovery_order)}
    inversions = 0
    for a in range(K):
        for b in range(a + 1, K):
            col_a_first = collapse_pos[a] < collapse_pos[b]
            rec_b_first = recovery_pos[b] < recovery_pos[a]
            if col_a_first != rec_b_first:
                inversions += 1

    return CollapseTrace(
        masks=masks.copy(),
        active=active.copy(),
        rd=rd.copy(),
        collapse_order=collapse_order,
        recovery_order=recovery_order,
        collapse_sig=">".join(str(k) for k in collapse_order),
        recovery_sig=">".join(str(k) for k in recovery_order),
        hysteresis_inversions=inversions,
    )


# ──────────────────────────────────────────────────────────────
# Volatility (A.90 §M2.12.12)
# ──────────────────────────────────────────────────────────────

def compute_state_volatility(
    trace: CollapseTrace,
    high_stress_rd_threshold: float = 0.45,
) -> Tuple[float, float]:
    """
    Returns (state_entropy, oscillation_rate).
    M2 targets: entropy < 1.0 nats, oscillation ≤ 0.10
    LSM targets: entropy > 1.5 nats, oscillation > 0.20
    """
    T   = len(trace.active)
    counts = np.zeros(NUM_STATES, dtype=np.float64)
    for s in trace.active:
        counts[int(s)] += 1.0
    probs   = safe_normalize(counts)
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

    hs_ticks = [t for t in range(T) if float(trace.rd[t]) >= high_stress_rd_threshold]
    if len(hs_ticks) < 2:
        osc_rate = 0.0
    else:
        transitions = sum(
            1 for i in range(1, len(hs_ticks))
            if trace.active[hs_ticks[i]] != trace.active[hs_ticks[i - 1]]
        )
        osc_rate = transitions / max(1, len(hs_ticks) - 1)

    return entropy, osc_rate


# ──────────────────────────────────────────────────────────────
# Sequence distribution utilities
# ──────────────────────────────────────────────────────────────

def aligned_distributions(
    sigs_a: List[str],
    sigs_b: List[str],
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    vocab = sorted(set(sigs_a) | set(sigs_b))
    ca, cb = Counter(sigs_a), Counter(sigs_b)
    pa = safe_normalize(np.array([float(ca[v]) for v in vocab]))
    pb = safe_normalize(np.array([float(cb[v]) for v in vocab]))
    return vocab, pa, pb


def average_pairwise_distance(sigs: List[str]) -> float:
    if len(sigs) < 2:
        return 0.0
    max_pairs = NUM_STATES * (NUM_STATES - 1) / 2.0

    def sig_to_pos(sig: str) -> Dict[int, int]:
        seq = [int(x) for x in sig.split(">")]
        return {s: i for i, s in enumerate(seq)}

    pos_maps = [sig_to_pos(s) for s in sigs]
    total, pairs = 0.0, 0
    for i in range(len(pos_maps)):
        for j in range(i + 1, len(pos_maps)):
            discord = sum(
                1 for a in range(NUM_STATES) for b in range(a + 1, NUM_STATES)
                if (pos_maps[i].get(a, 0) < pos_maps[i].get(b, 0)) !=
                   (pos_maps[j].get(a, 0) < pos_maps[j].get(b, 0))
            )
            total += discord / max_pairs
            pairs += 1
    return total / max(1, pairs)


# ──────────────────────────────────────────────────────────────
# Accessibility surface
# ──────────────────────────────────────────────────────────────

def compute_accessibility_surface(
    traces: List[CollapseTrace],
    rd_bin_edges: Optional[List[float]] = None,
) -> np.ndarray:
    """Returns [num_bins, K] P(accessible | rd_bin)."""
    if rd_bin_edges is None:
        rd_bin_edges = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.01]
    B = len(rd_bin_edges) - 1
    counts = np.zeros((B, NUM_STATES), dtype=np.float64)
    totals = np.zeros(B, dtype=np.float64)
    for trace in traces:
        for t in range(len(trace.rd)):
            rv = float(trace.rd[t])
            for b in range(B):
                if rd_bin_edges[b] <= rv < rd_bin_edges[b + 1]:
                    counts[b] += trace.masks[t].astype(float)
                    totals[b] += 1.0
                    break
    surface = np.zeros((B, NUM_STATES), dtype=np.float64)
    for b in range(B):
        if totals[b] > 0:
            surface[b] = counts[b] / totals[b]
    return surface


def accessibility_surface_kl(
    surf_m2: np.ndarray,
    surf_lsm: np.ndarray,
) -> float:
    B = surf_m2.shape[0]
    return sum(kl_divergence(surf_m2[b], surf_lsm[b]) for b in range(B)) / max(1, B)


# ──────────────────────────────────────────────────────────────
# Calibration: align LSM latent states → M2 families
# ──────────────────────────────────────────────────────────────

@dataclass
class CalibrationResult:
    perm_lsm_to_m2:       List[int]
    m2_state_action_dists: np.ndarray   # [K, A]
    lsm_state_action_dists: np.ndarray  # [K, A]


def estimate_state_action_dists(
    wrapper: AgentWrapper,
    seeds: List[int],
    num_steps: int,
    rd_value: float,
    num_actions: int,
) -> np.ndarray:
    counts = np.zeros((NUM_STATES, num_actions), dtype=np.float64)
    state_counts = np.zeros(NUM_STATES, dtype=np.float64)
    for seed in seeds:
        wrapper.reset(seed)
        for t in range(num_steps):
            obs = [rd_value] + [0.0] * 15
            out = wrapper.step(obs)
            z = int(out.active_state) % NUM_STATES
            logits = np.array(out.action_logits, dtype=np.float64)
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            counts[z] += probs
            state_counts[z] += 1.0
    for k in range(NUM_STATES):
        if state_counts[k] > 0:
            counts[k] /= state_counts[k]
        else:
            counts[k] = np.ones(num_actions) / num_actions
    return counts.astype(np.float32)


def calibrate(
    m2_wrapper: AgentWrapper,
    lsm_wrapper: AgentWrapper,
    seeds: List[int],
    num_steps: int = 80,
    rd_value: float = 0.10,
    num_actions: int = 128,
) -> CalibrationResult:
    m2_dists  = estimate_state_action_dists(m2_wrapper,  seeds, num_steps, rd_value, num_actions)
    lsm_dists = estimate_state_action_dists(lsm_wrapper, seeds, num_steps, rd_value, num_actions)
    cost = pairwise_js_matrix(lsm_dists, m2_dists)
    perm = best_permutation_min_cost(cost)
    return CalibrationResult(perm_lsm_to_m2=perm, m2_state_action_dists=m2_dists, lsm_state_action_dists=lsm_dists)


# ──────────────────────────────────────────────────────────────
# Episode runner
# ──────────────────────────────────────────────────────────────

def run_episode(
    wrapper: AgentWrapper,
    seed: int,
    schedule: CatastropheSchedule,
    perm: Optional[List[int]] = None,
) -> CollapseTrace:
    random.seed(seed)
    np.random.seed(seed)
    wrapper.reset(seed)

    rd_sched = schedule.build(seed=seed)
    T = len(rd_sched)
    masks_t  = np.zeros((T, NUM_STATES), dtype=np.int8)
    active_t = np.zeros(T, dtype=np.int32)

    for t in range(T):
        rd  = float(rd_sched[t])
        obs = [rd] + [0.0] * 15
        out = wrapper.step(obs)
        # Positions 0–6 of available_mask encode per-family accessibility (A(f,t) > 0)
        # This is set by M2MinimalPolicy.forward() — family_mask encoded at positions 0-6
        full_mask = out.available_mask
        if len(full_mask) >= NUM_STATES:
            mask = np.array(full_mask[:NUM_STATES], dtype=np.int8)
        else:
            mask = np.ones(NUM_STATES, dtype=np.int8)  # fallback: all accessible
        z    = int(out.active_state)

        if perm is not None:
            # Apply LSM→M2 permutation
            permuted_mask = np.zeros(NUM_STATES, dtype=np.int8)
            for src, tgt in enumerate(perm):
                if src < len(mask):
                    permuted_mask[tgt % NUM_STATES] = mask[src]
            mask = permuted_mask
            z = perm[z % NUM_STATES]

        masks_t[t]  = mask[:NUM_STATES] if len(mask) >= NUM_STATES else np.pad(mask, (0, NUM_STATES - len(mask)))
        active_t[t] = z % NUM_STATES

    return extract_collapse_trace(masks_t, active_t, rd_sched, schedule)


# ──────────────────────────────────────────────────────────────
# Topology metrics + results
# ──────────────────────────────────────────────────────────────

@dataclass
class TopologyMetrics:
    vocab:                       List[str]
    p_m2:                        np.ndarray
    p_lsm:                       np.ndarray
    js_m2_lsm:                   float      # target > 0.10 for battery win
    m2_seed_instability:         float
    lsm_seed_instability:        float
    m2_mean_spearman:            float      # target ≥ 0.85
    lsm_mean_spearman:           float
    accessibility_surface_kl:    float
    m2_mean_hysteresis_inversions: float
    lsm_mean_hysteresis_inversions: float
    m2_mean_volatility:          float      # target < 1.0 nats
    lsm_mean_volatility:         float      # target > 1.5 nats
    m2_mean_oscillation_rate:    float      # target ≤ 0.10
    lsm_mean_oscillation_rate:   float      # target > 0.20
    topology_win:                bool

    def summary(self) -> str:
        lines = ["\n══ Topology Suite ══════════════════════════════════════"]
        def row(label, m2v, lsmv, target=""):
            win = "✓" if m2v > lsmv else ("=" if abs(m2v - lsmv) < 0.005 else "✗")
            return f"  {win} {label:<35} M2={m2v:.4f}  LSM={lsmv:.4f}  {target}"
        def row_inv(label, m2v, lsmv, target=""):
            win = "✓" if m2v < lsmv else ("=" if abs(m2v - lsmv) < 0.005 else "✗")
            return f"  {win} {label:<35} M2={m2v:.4f}  LSM={lsmv:.4f}  {target}"
        lines.append(row("Spearman vs bio prior",   self.m2_mean_spearman, self.lsm_mean_spearman, "target M2≥0.85"))
        lines.append(f"  {'✓' if self.js_m2_lsm > 0.10 else '✗'} {'JS(M2,LSM) signature dist':<35} JS={self.js_m2_lsm:.4f}  target>0.10")
        lines.append(row_inv("Seed instability",    self.m2_seed_instability, self.lsm_seed_instability, "target M2<LSM"))
        lines.append(row_inv("State entropy",       self.m2_mean_volatility, self.lsm_mean_volatility, "M2<1.0 nats"))
        lines.append(row_inv("Oscillation rate",    self.m2_mean_oscillation_rate, self.lsm_mean_oscillation_rate, "M2≤0.10"))
        lines.append(f"\n  Topology Win: {'YES ✓' if self.topology_win else 'NO ✗'}")
        return "\n".join(lines)


@dataclass
class LatentVsM2TopologySuiteResult:
    calibration: CalibrationResult
    metrics:     TopologyMetrics
    m2_traces:   List[CollapseTrace]
    lsm_traces:  List[CollapseTrace]


def compute_topology_metrics(
    m2_traces: List[CollapseTrace],
    lsm_traces: List[CollapseTrace],
) -> TopologyMetrics:
    m2_sigs  = [t.collapse_sig for t in m2_traces]
    lsm_sigs = [t.collapse_sig for t in lsm_traces]

    vocab, p_m2, p_lsm = aligned_distributions(m2_sigs, lsm_sigs)
    js = js_divergence(p_m2, p_lsm)

    bio = list(BIOLOGICAL_COLLAPSE_PRIOR)
    def spearman_trace(trace):
        order = list(trace.collapse_order)
        bio_rank = [bio.index(k) if k in bio else len(bio) for k in range(NUM_STATES)]
        obs_rank = [order.index(k) if k in order else len(order) for k in range(NUM_STATES)]
        return spearman_corr(bio_rank, obs_rank)

    m2_spearman  = [spearman_trace(t) for t in m2_traces]
    lsm_spearman = [spearman_trace(t) for t in lsm_traces]

    m2_surf  = compute_accessibility_surface(m2_traces)
    lsm_surf = compute_accessibility_surface(lsm_traces)

    m2_vol  = [compute_state_volatility(t) for t in m2_traces]
    lsm_vol = [compute_state_volatility(t) for t in lsm_traces]

    def mean(lst): return sum(lst) / max(1, len(lst))

    m2_mean_vol  = mean([v[0] for v in m2_vol])
    lsm_mean_vol = mean([v[0] for v in lsm_vol])
    m2_mean_osc  = mean([v[1] for v in m2_vol])
    lsm_mean_osc = mean([v[1] for v in lsm_vol])

    topology_win = (
        m2_mean_vol < lsm_mean_vol
        and mean(m2_spearman) > mean(lsm_spearman)
        and js > 0.10
    )

    return TopologyMetrics(
        vocab=vocab, p_m2=p_m2, p_lsm=p_lsm,
        js_m2_lsm=js,
        m2_seed_instability=average_pairwise_distance(m2_sigs),
        lsm_seed_instability=average_pairwise_distance(lsm_sigs),
        m2_mean_spearman=mean(m2_spearman),
        lsm_mean_spearman=mean(lsm_spearman),
        accessibility_surface_kl=accessibility_surface_kl(m2_surf, lsm_surf),
        m2_mean_hysteresis_inversions=mean([t.hysteresis_inversions for t in m2_traces]),
        lsm_mean_hysteresis_inversions=mean([t.hysteresis_inversions for t in lsm_traces]),
        m2_mean_volatility=m2_mean_vol,
        lsm_mean_volatility=lsm_mean_vol,
        m2_mean_oscillation_rate=m2_mean_osc,
        lsm_mean_oscillation_rate=lsm_mean_osc,
        topology_win=topology_win,
    )


def test_latent_vs_m2_topology_suite(
    m2_wrapper:  AgentWrapper,
    lsm_wrapper: AgentWrapper,
    num_actions:        int        = 128,
    calibration_seeds:  List[int]  = None,
    test_seeds:         List[int]  = None,
    calibration_steps:  int        = 80,
    calibration_rd:     float      = 0.10,
    schedule: Optional[CatastropheSchedule] = None,
) -> LatentVsM2TopologySuiteResult:
    if calibration_seeds is None:
        calibration_seeds = list(range(8))
    if test_seeds is None:
        test_seeds = list(range(16))
    if schedule is None:
        schedule = CatastropheSchedule()

    cal = calibrate(m2_wrapper, lsm_wrapper, calibration_seeds, calibration_steps, calibration_rd, num_actions)
    m2_traces  = [run_episode(m2_wrapper,  s, schedule, perm=None)              for s in test_seeds]
    lsm_traces = [run_episode(lsm_wrapper, s, schedule, perm=cal.perm_lsm_to_m2) for s in test_seeds]
    metrics    = compute_topology_metrics(m2_traces, lsm_traces)
    print(metrics.summary())
    return LatentVsM2TopologySuiteResult(calibration=cal, metrics=metrics, m2_traces=m2_traces, lsm_traces=lsm_traces)
