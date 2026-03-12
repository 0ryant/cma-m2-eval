"""
regime1_runner.py  —  P3.1
============================
Full Regime 1 PEACETIME arc.

≥ 32 agents, ≥ 3 seeds, 500 ticks per episode.
Collects Observables 1–5 from the pre-registration doc.
Checks cold-start decay gate (P1.4) after each seed.
Writes per-tick telemetry for downstream battery and FMB suite.

Regime 1 parameters:
  - rd ramps from 0.0 → 0.40 over the arc (PEACETIME ceiling)
  - Stress shocks injected at ticks 150, 300 (brief STRESS spikes, rd → 0.55, 5 ticks)
  - Full recovery after each shock

§ references: CMA v4.1 roadmap §P3.1, M2 v1.18 §M2.12.11, §36.2 cold-start gate
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math
import random
import statistics
import numpy as np

from telemetry import (
    TelemetryEmitter, TickRecord, M2Family, M2Overlay,
    PrecedenceTag, check_depressive_lock_in,
)
from agent_wrapper import AgentWrapper, M2AgentWrapper
from replay_buffer import StratifiedReplayBuffer, ColdStartDecayGate, Transition, Regime
from baseline_suite import (
    tactic_class_cv, switch_frequency, oscillation_rate, spearman_collapse_rank,
    check_cold_start_gate,
)


# ──────────────────────────────────────────────────────────────
# Regime 1 config
# ──────────────────────────────────────────────────────────────

@dataclass
class Regime1Config:
    num_agents:          int   = 32
    num_seeds:           int   = 3
    ticks_per_episode:   int   = 500
    obs_dim:             int   = 16

    # Stress schedule (PEACETIME arc)
    rd_baseline:         float = 0.05
    rd_ramp_ceiling:     float = 0.40
    stress_shock_ticks:  List[int]  = field(default_factory=lambda: [150, 300])
    stress_shock_rd:     float = 0.55
    stress_shock_duration: int = 5

    # Cold-start gate
    gate_tick:           int   = 150   # world_model_error < 0.30 by this tick
    gate_threshold:      float = 0.30

    # Replay buffer
    replay_capacity:     int   = 1000
    replay_peacetime_fraction: float = 0.30

    # World model error decay (simulated — replace with real WM loss)
    wm_error_decay_rate: float = 0.005   # per tick


# ──────────────────────────────────────────────────────────────
# rd schedule builder
# ──────────────────────────────────────────────────────────────

def build_peacetime_schedule(
    cfg:  Regime1Config,
    seed: int,
) -> np.ndarray:
    """
    PEACETIME arc rd schedule.
    Linear ramp 0.05 → 0.40 with brief stress shocks injected.
    """
    rng = np.random.default_rng(seed)
    T   = cfg.ticks_per_episode
    rd  = np.linspace(cfg.rd_baseline, cfg.rd_ramp_ceiling, T).astype(np.float32)
    # Small baseline jitter
    rd += rng.normal(0, 0.01, size=T).astype(np.float32)

    # Stress shocks
    for shock_tick in cfg.stress_shock_ticks:
        for dt in range(cfg.stress_shock_duration):
            t = shock_tick + dt
            if 0 <= t < T:
                # ramp up then back down
                intensity = 1.0 - abs(dt - cfg.stress_shock_duration // 2) / (cfg.stress_shock_duration / 2)
                rd[t] = max(rd[t], cfg.stress_shock_rd * intensity)

    return np.clip(rd, 0.0, 1.0)


# ──────────────────────────────────────────────────────────────
# Observable collection
# ──────────────────────────────────────────────────────────────

@dataclass
class ObservableSnapshot:
    """Per-seed snapshot of all pre-registered observables."""
    seed:                    int

    # Observable 1: Strategic individuality
    cv_tactic_class:         float

    # Observable 2: Costly switching
    switch_frequency:        float
    post_switch_degradation: float   # relative perf drop in 5 ticks post-switch

    # Observable 3: Stress collapse order
    spearman_collapse_rho:   float

    # Observable 4: Legibility (collected by battery; stub here)
    social_signal_lift_m2:   float = 0.0
    social_signal_lift_lsm:  float = 0.0

    # Observable 5: Character formation
    pearson_r_char_stability: float = 0.0

    # Supplementary
    cold_start_gate_passed:  bool  = False
    mean_world_model_error_at_150: float = 1.0
    precedence_strong_fraction: float = 0.0
    depressive_lock_in_events:  int   = 0


def compute_post_switch_degradation(
    records: List[TickRecord],
    window:  int = 5,
) -> float:
    """
    Observable 2b: mean relative performance drop in window ticks after a switch,
    vs window ticks before. Uses narrative_coherence as proxy for performance
    (real engine: swap for reward or goal_progress).
    """
    pre_vals, post_vals = [], []
    for i, r in enumerate(records):
        if r.switch_cost_paid:
            pre  = [records[j].narrative_coherence for j in range(max(0, i - window), i)]
            post = [records[j].narrative_coherence for j in range(i + 1, min(len(records), i + window + 1))]
            if pre and post:
                pre_vals.append(sum(pre) / len(pre))
                post_vals.append(sum(post) / len(post))
    if not pre_vals:
        return 0.0
    degradation = [(pre - post) / max(pre, 1e-6) for pre, post in zip(pre_vals, post_vals)]
    return sum(degradation) / len(degradation)


def compute_character_stability(
    records_by_episode: List[List[TickRecord]],
    n_episodes: int = 3,
) -> float:
    """
    Observable 5: Pearson r between family transition matrix at episode N vs N+3.
    Returns 0.0 if insufficient episodes.
    """
    if len(records_by_episode) < n_episodes + 1:
        return 0.0

    def transition_matrix(records: List[TickRecord]) -> np.ndarray:
        K = len(M2Family) - 2  # exclude BASELINE and NONE
        T = np.zeros((K, K), dtype=np.float64)
        for i in range(1, len(records)):
            a = records[i - 1].active_policy_family
            b = records[i].active_policy_family
            ai = list(M2Family).index(a) % K
            bi = list(M2Family).index(b) % K
            T[ai, bi] += 1.0
        row_sums = T.sum(axis=1, keepdims=True)
        T = T / np.maximum(row_sums, 1.0)
        return T

    Ts = [transition_matrix(records_by_episode[i]) for i in range(min(len(records_by_episode), n_episodes + 2))]
    if len(Ts) < 2:
        return 0.0

    T0 = Ts[0].flatten()
    T3 = Ts[min(n_episodes, len(Ts) - 1)].flatten()

    # Pearson r
    if T0.std() < 1e-8 or T3.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(T0, T3)[0, 1])


# ──────────────────────────────────────────────────────────────
# Single-seed episode runner
# ──────────────────────────────────────────────────────────────

def run_seed(
    agents:  List[AgentWrapper],
    cfg:     Regime1Config,
    seed:    int,
    tel:     TelemetryEmitter,
    buf:     StratifiedReplayBuffer,
    gate:    ColdStartDecayGate,
    rng:     random.Random,
    env=None,   # SimEnv instance; created here if None
) -> Tuple[List[TickRecord], bool]:
    """
    Run all agents for one seed against real SimEnv. Returns (all_records, gate_passed).
    """
    from sim_env import SimEnv, EnvConfig

    all_records: List[TickRecord] = []

    # Build env from cfg dimensions if not provided
    if env is None:
        env_cfg = EnvConfig(
            n_agents=len(agents),
            obs_dim=cfg.obs_dim,
            ticks_per_episode=cfg.ticks_per_episode,
            seed=seed,
        )
        env = SimEnv(env_cfg, seed=seed)

    obs_dict = env.reset(seed=seed)
    for agent in agents:
        agent.reset(seed)

    # Map agent wrapper order to env agent IDs
    agent_ids = [getattr(a, 'agent_id', f'agent_{i}') for i, a in enumerate(agents)]

    for t in range(cfg.ticks_per_episode):
        tick_info = env.tick_info()
        pop_wm_error = env.population_mean_wm_error()
        pop_rd       = env.population_mean_rd()
        regime_label = Regime.PEACETIME if pop_rd < 0.50 else Regime.STRESS

        action_dict: dict = {}
        tick_records: List[TickRecord] = []

        for i, agent in enumerate(agents):
            aid = agent_ids[i]
            obs = obs_dict.get(aid, [0.0] * cfg.obs_dim)
            out = agent.step(obs)

            # Real env metrics from SimEnv
            info = tick_info.get(aid, {})
            rd   = info.get("rd",   obs[0] if obs else 0.0)
            nc   = info.get("narrative_coherence", 1.0)
            wme  = info.get("world_model_error", pop_wm_error)
            pgv  = info.get("primary_goal_valence", 0.0)

            # Map action index → tactic family name for env.step()
            family_val = list(M2Family)[out.active_state % 7] if out.active_state < 7 else M2Family.BASELINE
            action_name = family_val.value if family_val != M2Family.BASELINE else "BASELINE"
            action_dict[aid] = action_name

            # Policy score vector from M2MinimalPolicy explanation trace
            policy_scores = [0.0] * 7
            if hasattr(agent, 'policy_layer') and hasattr(agent.policy_layer, 'explanation_trace'):
                trace = agent.policy_layer.explanation_trace()
                if trace:
                    last = trace[-1]
                    family_order = ["DEFEND","WITHDRAW","REPAIR","EXPLORE","DOMINATE","SEEK_HELP","DECEIVE"]
                    for fi, fname in enumerate(family_order):
                        policy_scores[fi] = last.policy_scores.get(fname, 0.0)

            # Accessible families from policy
            accessible = [family_val]
            if hasattr(agent, 'policy_layer') and hasattr(agent.policy_layer, 'explanation_trace'):
                trace = agent.policy_layer.explanation_trace()
                if trace:
                    last = trace[-1]
                    accessible = [
                        M2Family[fname] if fname in M2Family.__members__ else M2Family.BASELINE
                        for fname in last.accessible_families
                    ] or [family_val]

            # Switch detection from policy trace
            switch_paid = False
            switch_mag  = 0.0
            if hasattr(agent, 'policy_layer') and hasattr(agent.policy_layer, 'explanation_trace'):
                trace = agent.policy_layer.explanation_trace()
                if trace:
                    last = trace[-1]
                    switch_paid = last.switch_occurred
                    switch_mag  = last.switch_cost_paid

            record = TickRecord(
                tick=t,
                agent_id=aid,
                regime=regime_label,
                seed=seed,
                active_policy_family=family_val,
                policy_score_vector=policy_scores,
                switch_cost_paid=switch_paid,
                switch_cost_magnitude=switch_mag,
                accessible_families=accessible,
                active_overlays=[],
                policy_conflict_detected=False,
                tactic_class=out.tactic_class,
                action_taken=out.action_taken,
                precedence_tag=out._precedence_tag if hasattr(out, '_precedence_tag') else PrecedenceTag.SCORE_WIN,
                dominant_module=out._dominant_module if hasattr(out, '_dominant_module') else "M2_policy",
                regression_depth=rd,
                baseline_ticks_running=agent._baseline_ticks if hasattr(agent, '_baseline_ticks') else 0,
                mourn_during_baseline_ticks=agent._mourn_ticks_in_baseline if hasattr(agent, '_mourn_ticks_in_baseline') else 0,
                narrative_coherence=nc,
                world_model_error=wme,
                primary_goal_valence=pgv,
                obs_raw=obs,   # stored for CF probe contexts in LSM battery (P2.4)
            )
            tel.tick(record)
            tick_records.append(record)

            # Replay buffer
            rd_next = max(0.0, rd - 0.01)
            buf.add(Transition([rd, nc, wme, pgv], out.action_taken,
                               [rd_next, nc, max(0.02, wme - 0.005), pgv],
                               regime_label, t, aid))

        all_records.extend(tick_records)

        # Advance env one tick
        obs_dict, done = env.step(action_dict)

        # Cold-start gate: track population mean world model error
        gate.record(t, pop_wm_error)

        if done:
            break

    # Check gate
    gate_passed, gate_msg = gate.check_regime_1_gate()
    gate_blocked = "not yet reached" in gate_msg or "cannot be evaluated" in gate_msg
    if gate_blocked:
        print(f"  Seed {seed} cold-start gate: BLOCKED — {gate_msg}")
    else:
        print(f"  Seed {seed} cold-start gate: {'PASS' if gate_passed else 'FAIL'} — {gate_msg}")
    gate_result = None if gate_blocked else gate_passed
    return all_records, gate_result


# ──────────────────────────────────────────────────────────────
# Full Regime 1 run
# ──────────────────────────────────────────────────────────────

@dataclass
class Regime1Result:
    observables:        List[ObservableSnapshot]
    all_records:        List[TickRecord]
    telemetry:          TelemetryEmitter
    gate_passed:        Optional[bool]  # True=all pass, False=any fail, None=blocked (run too short)
    regime2_unblocked:  bool            # True only if gate_passed is True

    def summary(self) -> str:
        lines = ["\n══ Regime 1 PEACETIME Arc ══════════════════════════════"]
        if not self.observables:
            return lines[0] + "\n  No data."
        obs = self.observables
        def mean(key): return sum(getattr(o, key) for o in obs) / len(obs)

        lines.append(f"  Seeds run:         {len(obs)}")
        lines.append(f"  Total tick records: {len(self.all_records):,}")
        lines.append(f"")
        lines.append(f"  Observable 1 — CV(tactic_class):     {mean('cv_tactic_class'):.4f}  (target M2 > 0.25)")
        lines.append(f"  Observable 2 — Switch freq:          {mean('switch_frequency'):.4f}")
        lines.append(f"  Observable 2 — Post-switch degr.:    {mean('post_switch_degradation'):.4f}  (target ≤ −0.10)")
        lines.append(f"  Observable 3 — Spearman collapse ρ:  {mean('spearman_collapse_rho'):.4f}  (target ≥ 0.85)")
        lines.append(f"  Observable 5 — Character stability:  {mean('pearson_r_char_stability'):.4f}  (target ≥ 0.70)")
        lines.append(f"")
        lines.append(f"  Precedence strong frac: {mean('precedence_strong_fraction'):.3f}  (target > 0.70 for Tier A)")
        if self.gate_passed is None:
            gate_str = "BLOCKED — run too short (ticks < 150)"
        elif self.gate_passed:
            gate_str = "ALL PASS ✓"
        else:
            gate_str = "FAIL ✗ — fix replay buffer before Regime 2"
        lines.append(f"  Cold-start gate:        {gate_str}")
        lines.append(f"  Regime 2 unblocked:     {'YES' if self.regime2_unblocked else 'NO'}")
        return "\n".join(lines)


def run_regime1(
    m2_agents:  List[AgentWrapper],
    cfg:        Regime1Config = None,
    seed_base:  int = 0,
) -> Regime1Result:
    """
    Full Regime 1 PEACETIME arc against real SimEnv.

    Args:
        m2_agents:  list of M2AgentWrapper instances (≥ 32 for publication)
        cfg:        Regime1Config
        seed_base:  starting seed (seeds = seed_base, seed_base+1, ...)
    """
    from sim_env import SimEnv, EnvConfig

    if cfg is None:
        cfg = Regime1Config()

    tel  = TelemetryEmitter()
    buf  = StratifiedReplayBuffer(cfg.replay_capacity, cfg.replay_peacetime_fraction)
    gate = ColdStartDecayGate()
    rng  = random.Random(seed_base)

    all_records:    List[TickRecord]        = []
    snapshots:      List[ObservableSnapshot] = []
    all_gates_pass  = True

    for seed_offset in range(cfg.num_seeds):
        seed = seed_base + seed_offset
        print(f"\n── Regime 1 seed {seed} ─────────────────────────────")

        # Build fresh env for each seed
        env_cfg = EnvConfig(
            n_agents=len(m2_agents),
            obs_dim=cfg.obs_dim,
            ticks_per_episode=cfg.ticks_per_episode,
            seed=seed,
        )
        env = SimEnv(env_cfg, seed=seed)

        records, gate_ok = run_seed(m2_agents, cfg, seed, tel, buf, gate, rng, env=env)
        all_records.extend(records)
        if gate_ok is False:
            all_gates_pass = False
            print(f"  ⚠  Gate failed for seed {seed}. Calling double_peacetime_fraction().")
            buf.double_peacetime_fraction()
        elif gate_ok is None:
            print(f"  ⚠  Gate blocked for seed {seed} (run too short). Not a failure.")

        # Build per-agent episode lists for character stability
        by_agent: Dict[str, List[TickRecord]] = {}
        for r in records:
            by_agent.setdefault(r.agent_id, []).append(r)
        all_agent_episodes = list(by_agent.values())

        # Compute observables
        # Agents emit PRECEDENCE_TAG events to their shared tel (injected at build time).
        # The local tel collects tick records only. Pull precedence from agents shared tel.
        agent_tel = next((getattr(a, 'tel', None) for a in m2_agents if getattr(a, 'tel', None) is not None), tel)
        prec_frac = agent_tel.precedence_fraction()
        lock_ins  = sum(1 for e in agent_tel.event_log if hasattr(e, 'event_type') and getattr(e, 'event_type', '') == 'DEPRESSIVE_LOCK_IN')

        snap = ObservableSnapshot(
            seed=seed,
            cv_tactic_class=tactic_class_cv(records),
            switch_frequency=switch_frequency(records),
            post_switch_degradation=compute_post_switch_degradation(records),
            spearman_collapse_rho=spearman_collapse_rank(records),
            cold_start_gate_passed=gate_ok,
            mean_world_model_error_at_150=gate.error_by_tick.get(150, 1.0),
            precedence_strong_fraction=prec_frac.get('strong_fraction', 0.0),
            depressive_lock_in_events=lock_ins,
            pearson_r_char_stability=compute_character_stability(all_agent_episodes),
        )
        snapshots.append(snap)
        print(f"  CV(tactic): {snap.cv_tactic_class:.4f}  Spearman: {snap.spearman_collapse_rho:.4f}  "
              f"Pearson r: {snap.pearson_r_char_stability:.4f}")

    # Use agents shared telemetry (has PRECEDENCE_TAG events) for Tier A gate
    agent_tel = next((getattr(a, 'tel', None) for a in m2_agents if getattr(a, 'tel', None) is not None), tel)

    # Determine aggregate gate state:
    # all True → all_gates_pass=True, regime2_unblocked=True
    # any False → all_gates_pass=False, regime2_unblocked=False
    # all None (blocked) → gate_passed=None, regime2_unblocked=False
    gate_states = [snap.cold_start_gate_passed for snap in snapshots]
    if all(g is None for g in gate_states):
        final_gate = None
    elif all(g is True for g in gate_states if g is not None):
        final_gate = all_gates_pass  # True unless any False
    else:
        final_gate = False

    result = Regime1Result(
        observables=snapshots,
        all_records=all_records,
        telemetry=agent_tel,
        gate_passed=final_gate,
        regime2_unblocked=final_gate is True,
    )
    print(result.summary())
    return result
