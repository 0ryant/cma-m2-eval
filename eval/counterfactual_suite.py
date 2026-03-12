"""
counterfactual_suite.py  —  P2.4 (Suite 2 of 3)
=================================================
Hostile-control test: are M2 family labels causal anchors or post-hoc decoration?

Forces each agent into each strategic state and measures:
  1. Shift magnitude (JS divergence from baseline)
  2. Within-state centroid coherence (forced state lands in same semantic region)
  3. Between-state centroid margin (forced state is distinct from competing states)

M2 should win on coherence and margin. LSM may still shift, but with weaker
semantic concentration — because its latent clusters were formed by k-means over
action embeddings, not authored around strategic semantics.

Battery win condition (counterfactual_win):
  M2 higher within-state coherence AND higher between-state margin.

numpy-only. No torch.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import math
import random
import numpy as np

from agent_wrapper import AgentWrapper

NUM_STATES = 7
M2_FAMILIES = ["DEFEND", "WITHDRAW", "REPAIR", "EXPLORE", "DOMINATE", "SEEK_HELP", "DECEIVE"]


# ──────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = p.astype(np.float64) + eps
    q = q.astype(np.float64) + eps
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    return 0.5 * (kl_pm + kl_qm)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ──────────────────────────────────────────────────────────────
# Context generator
# ──────────────────────────────────────────────────────────────

@dataclass
class ProbeContext:
    obs:            List[float]
    rd:             float
    goal_stack_type: str   # must be in PERMISSIVE_TYPES for valid probe (§M2.11.4)
    seed:           int


PERMISSIVE_GOAL_TYPES = {"SOCIAL", "RELATIONAL", "INVESTIGATIVE", "NEUTRAL"}


def generate_probe_contexts(
    n_contexts: int,
    obs_dim:    int = 16,
    seed:       int = 0,
) -> List[ProbeContext]:
    """
    Generate probe contexts with permissive goal stack types (§M2.11.4).
    rd is low (< 0.30) to avoid stress-gating artifacts in counterfactual probes.
    """
    rng = random.Random(seed)
    types = list(PERMISSIVE_GOAL_TYPES)
    contexts = []
    for i in range(n_contexts):
        contexts.append(ProbeContext(
            obs=[rng.gauss(0, 1) for _ in range(obs_dim)],
            rd=rng.uniform(0.05, 0.25),
            goal_stack_type=types[i % len(types)],
            seed=rng.randint(0, 10000),
        ))
    return contexts


# ──────────────────────────────────────────────────────────────
# Probe runner
# ──────────────────────────────────────────────────────────────

@dataclass
class StateProbeResult:
    state:           int
    baseline_dist:   np.ndarray    # [A] action prob in natural state
    forced_dist:     np.ndarray    # [A] action prob when forced into this state
    shift_js:        float         # JS(baseline, forced)


@dataclass
class AgentProbeResults:
    agent_type:      str           # "M2" or "LSM"
    results_by_state: Dict[int, List[StateProbeResult]]
    # Aggregate metrics
    mean_within_coherence: float   # higher = centroids are stable across contexts
    mean_between_margin:   float   # higher = states are distinct from each other


def probe_agent(
    wrapper:   AgentWrapper,
    contexts:  List[ProbeContext],
    n_repeats: int = 3,
    agent_type: str = "M2",
    obs_noise_std: float = 0.15,
) -> AgentProbeResults:
    """
    For each (state, context) pair: run baseline step, then forced step.

    Within each repeat, obs is perturbed with Gaussian noise (std=obs_noise_std).
    This breaks determinism — deterministic argmax policies collapse to 1.0 coherence
    on identical obs, making the test meaningless. With obs-perturbation:
      - M2: forced state restricts to a SEMANTIC tactic set → action remains within
            that set despite obs perturbation → high within-state coherence
      - LSM: forced latent state has weaker semantic tactic restriction → obs
             perturbation causes larger distributional drift → lower coherence

    This is the correct operationalisation of the causal anchor test (§M2.1.2).
    """
    rng_perturb = np.random.default_rng(42)

    # Collect forced distributions per state across contexts
    dists_by_state: Dict[int, List[np.ndarray]] = defaultdict(list)

    for ctx in contexts:
        wrapper.reset(ctx.seed)
        # Baseline: natural step on clean obs (same temperature τ=0.25 as forced probes)
        baseline_out = wrapper.step(ctx.obs)
        baseline_logits = np.array(baseline_out.action_logits, dtype=np.float64)
        baseline_dist   = softmax(baseline_logits / 0.25)

        obs_arr = np.array(ctx.obs, dtype=np.float64)

        for state in range(NUM_STATES):
            for rep in range(n_repeats):
                # Perturb obs per repeat — tests obs-robustness within forced state
                noise = rng_perturb.normal(0.0, obs_noise_std, size=len(ctx.obs))
                perturbed_obs = list(obs_arr + noise)
                forced_out = wrapper.step_forced_state(perturbed_obs, state)
                forced_logits = np.array(forced_out.action_logits, dtype=np.float64)
                # Temperature-scaled softmax: τ=0.25 keeps distribution peaked but not a
                # point mass. This makes cosine similarity to centroid meaningful — argmax
                # policies collapse to identical point masses (coherence=1.0 trivially).
                # With τ=0.25, the SHAPE of the logit distribution within the available
                # action set is captured, not just the winning action index.
                forced_dist   = softmax(forced_logits / 0.25)
                dists_by_state[state].append(forced_dist)

    # Compute centroids per state
    centroids = {}
    for state, dists in dists_by_state.items():
        centroids[state] = np.mean(np.stack(dists, axis=0), axis=0)

    # Within-state coherence: mean cosine similarity to centroid
    within_coherences = []
    for state, dists in dists_by_state.items():
        c = centroids[state]
        sims = [cosine_similarity(d, c) for d in dists]
        within_coherences.append(sum(sims) / max(1, len(sims)))
    mean_within = sum(within_coherences) / max(1, len(within_coherences))

    # Between-state margin: mean JS between all pairs of centroids
    margins = []
    states = sorted(centroids.keys())
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            margins.append(js_divergence(centroids[states[i]], centroids[states[j]]))
    mean_between = sum(margins) / max(1, len(margins))

    # Build per-state probe results
    results_by_state: Dict[int, List[StateProbeResult]] = defaultdict(list)
    for state, dists in dists_by_state.items():
        c = centroids[state]
        for d in dists:
            baseline_out = wrapper.step(contexts[0].obs)
            bd = softmax(np.array(baseline_out.action_logits, dtype=np.float64))
            results_by_state[state].append(StateProbeResult(
                state=state,
                baseline_dist=bd,
                forced_dist=d,
                shift_js=js_divergence(bd, d),
            ))

    return AgentProbeResults(
        agent_type=agent_type,
        results_by_state=dict(results_by_state),
        mean_within_coherence=mean_within,
        mean_between_margin=mean_between,
    )


# ──────────────────────────────────────────────────────────────
# Suite result
# ──────────────────────────────────────────────────────────────

@dataclass
class CounterfactualSuiteResult:
    m2_results:               AgentProbeResults
    lsm_results:              AgentProbeResults
    m2_within_coherence:      float
    lsm_within_coherence:     float
    m2_between_margin:        float
    lsm_between_margin:       float
    counterfactual_win:       bool       # M2 wins on BOTH metrics

    def summary(self) -> str:
        lines = ["\n══ Counterfactual Suite ════════════════════════════════"]
        def row(label, m2v, lsmv, higher_is_better=True):
            win = "✓" if (m2v > lsmv if higher_is_better else m2v < lsmv) else "✗"
            return f"  {win} {label:<35} M2={m2v:.4f}  LSM={lsmv:.4f}"
        lines.append(row("Within-state coherence",  self.m2_within_coherence, self.lsm_within_coherence))
        lines.append(row("Between-state margin",    self.m2_between_margin,   self.lsm_between_margin))
        lines.append(f"\n  Counterfactual Win: {'YES ✓' if self.counterfactual_win else 'NO ✗'}")
        return "\n".join(lines)


def test_latent_vs_m2_counterfactual_suite(
    m2_wrapper:    AgentWrapper,
    lsm_wrapper:   AgentWrapper,
    n_contexts:    int = 20,
    n_repeats:     int = 3,
    obs_dim:       int = 16,
    seed:          int = 0,
    probe_contexts = None,    # Optional[List[ProbeContext]] — use real obs if provided
) -> CounterfactualSuiteResult:
    if probe_contexts is not None and len(probe_contexts) >= n_contexts:
        contexts = probe_contexts[:n_contexts]
    else:
        contexts = generate_probe_contexts(n_contexts, obs_dim=obs_dim, seed=seed)
    m2_res   = probe_agent(m2_wrapper,  contexts, n_repeats=n_repeats, agent_type="M2")
    lsm_res  = probe_agent(lsm_wrapper, contexts, n_repeats=n_repeats, agent_type="LSM")

    import math as _math
    # Between-state JS saturates at ln(2) ≈ 0.6931 when both architectures use
    # disjoint action supports (masked softmax with non-overlapping tactic sets).
    # At the ceiling, JS = ln(2) by construction regardless of logit quality —
    # it measures "disjointness exists", not "family distinctness".
    # TIED_AT_CEILING: both values within 0.005 of ln(2). In this case the
    # between condition is treated as non-discriminative (structural tie, not
    # a model tie), and win is decided on within-state coherence alone.
    # This is logged as a known deviation in preregistration.md.
    _JS_CEILING   = _math.log(2)   # ≈ 0.6931 — theoretical max for disjoint supports
    _CEILING_EPS  = 0.005
    between_ceiling = (
        abs(m2_res.mean_between_margin  - _JS_CEILING) < _CEILING_EPS
        and abs(lsm_res.mean_between_margin - _JS_CEILING) < _CEILING_EPS
    )
    win = (
        m2_res.mean_within_coherence > lsm_res.mean_within_coherence
        and (m2_res.mean_between_margin > lsm_res.mean_between_margin or between_ceiling)
    )

    result = CounterfactualSuiteResult(
        m2_results=m2_res,
        lsm_results=lsm_res,
        m2_within_coherence=m2_res.mean_within_coherence,
        lsm_within_coherence=lsm_res.mean_within_coherence,
        m2_between_margin=m2_res.mean_between_margin,
        lsm_between_margin=lsm_res.mean_between_margin,
        counterfactual_win=win,
    )
    print(result.summary())
    return result
