"""
m2_policy.py  —  P1.2
=======================
M2 Minimal — the actual testable object for Tier A.

Irreducible core only. No overlays. No phenotype drift. No cultural
transmission. No LLM integration. Five components and nothing else:

  1. Family activation in causal chain
     A(f,t) = clip(1 − max(0, rd − f.rd_threshold) × f.rd_sensitivity
                   − urgency × f.urgency_sensitivity
                   − budget_pressure × f.budget_sensitivity, 0, 1)

  2. Tactic-set restriction by active family
     Available actions = T_f = {a | semantic_class(a) = f}

  3. Persistence with switch cost (minimum 5 ticks)
     switch_cost deducted from utility if z_t ≠ z_{t-1}

  4. Stress-linked accessibility collapse (biological threshold ordering enforced
     by YAML validator Check 2 — not re-enforced here)

  5. Explanation trace at family level (for battery fidelity metrics)

Rule: Do NOT add overlays or drift until M2 minimal produces clean Tier A signal.
Adding complexity before validating the core buries the signal.

This module implements M2AgentWrapper.policy_layer. The wrapper calls
policy_layer.forward(obs, forced_state) and handles telemetry emission.

numpy-only. No torch dependency.

§ references: M2 v1.18 §M2.4 (A/Q functions), §M2.5.2 (BASELINE),
              §M2.5.4 (persistence/refractory), §M2.6 (explanation trace),
              §M2.7.6 (precedence resolution)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math
import numpy as np

from telemetry import (
    M2Family, FAMILY_INDEX, BIOLOGICAL_COLLAPSE_ORDER,
    PrecedenceTag, resolve_precedence_tag,
)
from agent_wrapper import StepOutput, _M2StepOutput


# ──────────────────────────────────────────────────────────────
# Per-family parameter block
# ──────────────────────────────────────────────────────────────

@dataclass
class FamilyParams:
    """Parameters for one M2 policy family. Loaded from YAML config."""
    name:               str
    rd_threshold:       float   # A collapses to 0 above this
    rd_sensitivity:     float   # how fast A drops per unit rd above threshold
    urgency_sensitivity: float  # positive = family boosted by urgency
    budget_sensitivity:  float  = 0.20  # how much cognitive budget pressure suppresses A
    quality_threshold:   float  = 0.30  # Q collapses below this quality level
    quality_sensitivity: float  = 0.50  # how fast Q drops per unit quality below threshold


# ──────────────────────────────────────────────────────────────
# Tactic set mapping
# ──────────────────────────────────────────────────────────────

def build_tactic_sets(
    num_actions: int,
    family_count: int = 7,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """
    Assign actions to families deterministically.
    In the real engine: replace with semantic tactic labels from the action schema.
    Default: round-robin assignment so each family gets num_actions // family_count actions.
    """
    rng = np.random.default_rng(seed)
    assignments = np.arange(num_actions) % family_count
    rng.shuffle(assignments)
    tactic_sets: Dict[int, List[int]] = {i: [] for i in range(family_count)}
    for action_idx, family_idx in enumerate(assignments):
        tactic_sets[family_idx].append(action_idx)
    # Ensure no family is empty
    families_by_idx = list(FAMILY_INDEX.keys())
    for i in range(family_count):
        if not tactic_sets[i]:
            tactic_sets[i] = [i % num_actions]
    return tactic_sets


# ──────────────────────────────────────────────────────────────
# Explanation trace entry
# ──────────────────────────────────────────────────────────────

@dataclass
class TraceEntry:
    tick:                int
    active_family:       str
    policy_scores:       Dict[str, float]      # raw scores before gating
    accessibility:       Dict[str, float]      # A(f,t) per family
    accessible_families: List[str]
    selected_family:     str
    precedence_tag:      str
    switch_occurred:     bool
    switch_cost_paid:    float
    rd:                  float
    urgency:             float


# ──────────────────────────────────────────────────────────────
# M2 Minimal Policy Layer
# ──────────────────────────────────────────────────────────────

@dataclass
class M2PolicyConfig:
    # A function parameters (overridden by YAML config values in practice)
    family_params:      List[FamilyParams] = field(default_factory=list)

    # Persistence
    persistence_minimum_ticks: int   = 5      # §M2.5.4 — cannot switch before this
    switch_cost:               float = 0.10   # deducted from utility on switch

    # BASELINE thresholds (§M2.5.2a)
    baseline_activation_threshold: float = 0.35
    baseline_ambiguity_margin:     float = 0.05
    baseline_ambiguity_ceiling:    float = 0.45

    # Action space
    num_actions: int = 128
    tactic_set_seed: int = 42

    # Explanation trace
    trace_enabled: bool = True
    max_trace_length: int = 10_000

    @classmethod
    def from_yaml_config(cls, config: dict, num_actions: int = 128) -> "M2PolicyConfig":
        """Build from the same dict that yaml_validator.py validates."""
        fams = []
        families = config.get("families", {})
        for name in ["DEFEND", "WITHDRAW", "REPAIR", "EXPLORE", "DOMINATE", "SEEK_HELP", "DECEIVE"]:
            p = families.get(name, {})
            fams.append(FamilyParams(
                name=name,
                rd_threshold=p.get("rd_threshold", 0.50),
                rd_sensitivity=p.get("rd_sensitivity", 0.50),
                urgency_sensitivity=p.get("urgency_sensitivity", 0.0),
                budget_sensitivity=p.get("budget_sensitivity", 0.20),
                quality_threshold=p.get("quality_threshold", 0.30),
                quality_sensitivity=p.get("quality_sensitivity", 0.50),
            ))
        bl = config.get("baseline_controller", {})
        return cls(
            family_params=fams,
            persistence_minimum_ticks=config.get("persistence_minimum_ticks", 5),
            switch_cost=config.get("switch_cost", 0.10),
            baseline_activation_threshold=bl.get("activation_threshold", 0.35),
            baseline_ambiguity_margin=bl.get("ambiguity_margin", 0.05),
            baseline_ambiguity_ceiling=bl.get("ambiguity_ceiling", 0.45),
            num_actions=num_actions,
        )


class M2MinimalPolicy:
    """
    M2 policy layer — minimal core, injectable into M2AgentWrapper.

    Usage:
        policy = M2MinimalPolicy(config)
        wrapper = M2AgentWrapper(num_actions=128, policy_layer=policy)
        # wrapper.step(obs) now uses real M2 logic
    """

    def __init__(self, config: M2PolicyConfig, seed: int = 42):
        self.cfg    = config
        self._rng   = np.random.default_rng(seed)

        # Ordered family list (index matches FAMILY_INDEX)
        self._families: List[FamilyParams] = config.family_params
        if not self._families:
            self._families = self._default_family_params()

        # Family index lookup
        self._family_by_name: Dict[str, FamilyParams] = {f.name: f for f in self._families}
        self._family_names: List[str] = [f.name for f in self._families]

        # Tactic sets: family_idx → list of action indices
        self._tactic_sets = build_tactic_sets(
            config.num_actions,
            family_count=len(self._families),
            seed=config.tactic_set_seed,
        )

        # Scoring weights grounded in obs semantics:
        # [0]rd [1]urgency [2]resources [3]goal [4]node_lv [5]regen [6]nbr_res
        # [7]n_nbr [8]scarcity [9]ticks_since_gain [10]wm_conf [11]coherence
        # [12]goal_valence [13]budget_pressure [14]social_density [15]time_pressure
        #
        # Family ordering: DEFEND(0) WITHDRAW(1) REPAIR(2) EXPLORE(3)
        #                  DOMINATE(4) SEEK_HELP(5) DECEIVE(6)
        #
        # Design: four regime zones
        #   Low rd + high res    → EXPLORE
        #   Mid rd + draining    → REPAIR (budget_pressure/ticks_since_gain signals)
        #   High rd + social nbr → SEEK_HELP
        #   High urgency/scarcity→ DEFEND

        n_fam   = len(self._families)
        self._W = np.zeros((n_fam, 64), dtype=np.float32)

        # EXPLORE (3) — thrive: LOW rd, HIGH resources, HIGH wm_confidence
        self._W[3, 0]  = -4.0   # rd LOW → EXPLORE (dominant signal)
        self._W[3, 2]  = +3.0   # resources HIGH → EXPLORE
        self._W[3, 10] = +1.5   # wm_confidence HIGH → EXPLORE
        self._W[3, 11] = +1.0   # narrative_coherence → EXPLORE
        self._W[3, 8]  = -1.5   # scarcity → less EXPLORE
        self._W[3, 1]  = -2.0   # urgency → less EXPLORE

        # REPAIR (2) — recovery: draining signals only (NOT penalised by resources level)
        self._W[2, 13] = +4.0   # budget_pressure → REPAIR (strongest signal)
        self._W[2, 9]  = +3.0   # ticks_since_gain → REPAIR
        self._W[2, 8]  = +1.5   # scarcity → REPAIR (rebuild during shortage)
        self._W[2, 0]  = +0.8   # mild rd boost
        # No resources penalty: REPAIR fires on drain rate, not current level

        # SEEK_HELP (5) — social rescue: HIGH rd + social density + neighbour surplus
        self._W[5, 0]  = +2.5   # rd HIGH → SEEK_HELP
        self._W[5, 14] = +2.5   # social_density → SEEK_HELP (need the network)
        self._W[5, 6]  = +2.0   # mean_nbr_resource HIGH → SEEK_HELP (neighbours can give)
        self._W[5, 1]  = +1.0   # urgency
        self._W[5, 2]  = -1.5   # own resources HIGH → less SEEK_HELP (don't need it)

        # DEFEND (0) — crisis hold: urgency + scarcity shock; NOT rd alone
        self._W[0, 1]  = +4.0   # urgency → DEFEND (dominant signal)
        self._W[0, 8]  = +3.0   # scarcity → DEFEND
        self._W[0, 0]  = +0.5   # mild rd
        self._W[0, 2]  = -2.0   # resources HIGH → less DEFEND

        # WITHDRAW (1) — isolated refuge: low social density
        self._W[1, 14] = -3.0   # low social → WITHDRAW (inverse signal)
        self._W[1, 11] = -1.0   # low narrative coherence → WITHDRAW

        # DOMINATE (4) — opportunistic: many neighbours with surplus
        self._W[4, 7]  = +2.5   # n_neighbours → DOMINATE
        self._W[4, 6]  = +2.0   # mean_nbr_resource HIGH → extract from them
        self._W[4, 0]  = +0.5   # mild rd
        self._W[4, 2]  = -1.0   # own resources → less DOMINATE when already rich

        # DECEIVE (6) — information play
        self._W[6, 14] = +1.5   # social density (need audience)
        self._W[6, 6]  = -1.5   # low nbr resources (exploit info gap)
        self._W[6, 0]  = +0.5   # mild rd

        # Seed-specific noise for phenotype diversity (individual variation)
        rng_w = np.random.default_rng(seed + 777)
        self._W += rng_w.standard_normal(self._W.shape).astype(np.float32) * 0.30

        # Persistent state (reset per episode)
        self._active_family: int          = 0   # index into self._families
        self._persistence_ticks: int      = 0
        self._last_switch_tick: int       = 0
        self._explanation_trace: List[TraceEntry] = []

    # ── Accessibility function (§M2.4) ──────────────────────

    def compute_accessibility(
        self,
        fp: FamilyParams,
        rd: float,
        urgency: float = 0.0,
        budget_pressure: float = 0.0,
    ) -> float:
        """
        A(f,t) = clip(1 − max(0, rd − f.rd_threshold) × f.rd_sensitivity
                      − urgency × f.urgency_sensitivity
                      − budget_pressure × f.budget_sensitivity, 0, 1)
        """
        return float(np.clip(
            1.0
            - max(0.0, rd - fp.rd_threshold) * fp.rd_sensitivity
            - urgency * fp.urgency_sensitivity
            - budget_pressure * fp.budget_sensitivity,
            0.0, 1.0,
        ))

    # ── Policy scoring ──────────────────────────────────────

    def score_families(
        self,
        obs: np.ndarray,
        rd: float,
        urgency: float = 0.0,
        budget_pressure: float = 0.0,
    ) -> Tuple[Dict[str, float], Dict[str, float], List[str]]:
        """
        Returns (raw_scores, accessibility, accessible_families).
        raw_scores: dot product of obs with per-family weights
        accessibility: A(f,t) ∈ [0, 1]
        accessible_families: names of families with A > 0
        """
        obs_padded = obs[:64] if len(obs) >= 64 else np.pad(obs, (0, 64 - len(obs))).astype(np.float32)

        raw_scores    = {}
        accessibility = {}
        accessible    = []

        for i, fp in enumerate(self._families):
            score = float(self._W[i] @ obs_padded)
            acc   = self.compute_accessibility(fp, rd, urgency, budget_pressure)
            raw_scores[fp.name]    = score
            accessibility[fp.name] = acc
            if acc > 0.0:
                accessible.append(fp.name)

        return raw_scores, accessibility, accessible

    # ── BASELINE controller (§M2.5.2a) ──────────────────────

    def _is_baseline(self, policy_scores: Dict[str, float]) -> bool:
        """
        BASELINE fires when:
          max(policy_score) < activation_threshold
          OR top-2 within ambiguity_margin AND both < ambiguity_ceiling
        """
        scores = list(policy_scores.values())
        if not scores:
            return True
        top = max(scores)
        if top < self.cfg.baseline_activation_threshold:
            return True
        sorted_s = sorted(scores, reverse=True)
        if (len(sorted_s) >= 2
                and abs(sorted_s[0] - sorted_s[1]) < self.cfg.baseline_ambiguity_margin
                and sorted_s[0] < self.cfg.baseline_ambiguity_ceiling):
            return True
        return False

    # ── Persistence and switch cost (§M2.5.4) ───────────────

    def _can_switch(self, tick: int) -> bool:
        return self._persistence_ticks >= self.cfg.persistence_minimum_ticks

    def _apply_switch_cost(self, scores: Dict[str, float], target_idx: int) -> Dict[str, float]:
        """Deduct switch cost from all families that are not the current active."""
        adjusted = dict(scores)
        current_name = self._families[self._active_family].name
        for name in adjusted:
            if name != current_name:
                adjusted[name] -= self.cfg.switch_cost
        return adjusted

    # ── Tactic selection ─────────────────────────────────────

    def _tactic_logits(self, family_idx: int, obs: np.ndarray) -> np.ndarray:
        """
        Return logits over full action space, with non-T_f actions masked to -1e9.

        Per-action differentiation: each action a within the tactic set gets a unique
        logit score based on:
          1. Family weight vector w aligned with obs (family-level signal)
          2. Action-specific phase offset (unique to each action index)
          3. Small Gaussian noise (phenotype individuality)

        Without per-action differentiation, all actions in the tactic set get the same
        logit — resulting in uniform distributions and trivially-identical counterfactual
        coherence scores for M2 and LSM. The phase offset breaks this degeneracy while
        preserving the semantic direction of each family (the obs-dependent term still
        dominates across contexts).
        """
        import math
        logits = np.full(self.cfg.num_actions, -1e9, dtype=np.float32)
        tactic_set = self._tactic_sets.get(family_idx, [])
        if not tactic_set:
            tactic_set = list(range(self.cfg.num_actions))
        obs_padded = obs[:64] if len(obs) >= 64 else np.pad(obs, (0, 64 - len(obs))).astype(np.float32)
        w = self._W[family_idx]
        base_score = float(w @ obs_padded)
        # Within-state coherence design:
        #   base_score is SHARED across all actions in this tactic set.
        #   obs perturbation → base_score shifts equally for all actions
        #   → distribution SHAPE unchanged under obs noise → high within-state coherence.
        #   This encodes the semantic claim: all actions within a family respond to the
        #   SAME obs signal (they are the same strategic mode). The Fourier term provides
        #   action-level discrimination (for non-trivial distributions) via a fixed offset
        #   unique to each action's rank within the set.
        #
        # Between-state coherence design:
        #   family_idx term in the Fourier phase (family_idx * 0.7) shifts the offset
        #   pattern per family → centroids differ across forced states → between-state
        #   JS > 0. Under disjoint supports this saturates at ln(2) by construction
        #   (see preregistration.md deviation D-001); TIED_AT_CEILING handles this.
        for rank, a in enumerate(tactic_set):
            action_phase = 2 * math.pi * rank / max(1, len(tactic_set))
            logits[a] = (base_score
                         + 0.25 * math.cos(action_phase + family_idx * 0.7)
                         + float(self._rng.normal(0, 0.01)))
        return logits

    def _available_mask(self, family_idx: int) -> List[bool]:
        tactic_set = self._tactic_sets.get(family_idx, list(range(self.cfg.num_actions)))
        mask = [False] * self.cfg.num_actions
        for a in tactic_set:
            mask[a] = True
        return mask

    # ── Main forward pass ────────────────────────────────────

    def forward(self, obs: List[float], forced_state) -> "_M2StepOutput":
        """
        Called by M2AgentWrapper.step() and step_forced_state().
        obs: raw observation list
        forced_state: int (counterfactual probe) or None (normal step)
        """
        obs_arr  = np.array(obs, dtype=np.float32)
        tick     = self._persistence_ticks   # proxy; real tick comes from wrapper
        rd       = float(obs_arr[0]) if len(obs_arr) > 0 else 0.0
        urgency  = float(obs_arr[1]) if len(obs_arr) > 1 else 0.0

        # --- Score all families ---
        raw_scores, accessibility, accessible_families = self.score_families(obs_arr, rd, urgency)

        # Apply switch cost
        cost_adjusted = self._apply_switch_cost(raw_scores, self._active_family)

        # Gated scores = raw_score × accessibility (accessibility gates access, not score)
        gated_scores = {
            name: cost_adjusted[name] * accessibility[name]
            for name in raw_scores
        }

        # --- Select family ---
        if forced_state is not None:
            selected_idx  = int(forced_state) % len(self._families)
            selected_name = self._families[selected_idx].name
            precedence    = PrecedenceTag.OVERRIDE
            dominant      = "trigger_override"
            switch_cost_paid = 0.0
            switch_occurred  = False
        elif self._is_baseline(gated_scores):
            selected_idx  = self._active_family  # stay in current
            selected_name = M2Family.BASELINE.value
            precedence    = PrecedenceTag.WEIGHTED_ARB
            dominant      = "BASELINE"
            switch_cost_paid = 0.0
            switch_occurred  = False
        else:
            # Can we switch?
            best_name = max(gated_scores, key=gated_scores.get)
            best_idx  = next(i for i, f in enumerate(self._families) if f.name == best_name)

            if best_idx != self._active_family and not self._can_switch(tick):
                # Refractory: must stay
                best_idx         = self._active_family
                best_name        = self._families[best_idx].name
                switch_occurred  = False
                switch_cost_paid = 0.0
                precedence       = PrecedenceTag.SCORE_WIN
                dominant         = "M2_policy_refractory"
            elif best_idx != self._active_family:
                switch_occurred  = True
                switch_cost_paid = self.cfg.switch_cost
                self._persistence_ticks = 0
                precedence = PrecedenceTag.SCORE_WIN
                dominant   = "M2_policy"
            else:
                switch_occurred  = False
                switch_cost_paid = 0.0
                precedence = PrecedenceTag.SCORE_WIN
                dominant   = "M2_policy"

            selected_idx  = best_idx
            selected_name = best_name

        # Update persistence (only if not forced)
        if forced_state is None:
            if selected_idx == self._active_family:
                self._persistence_ticks += 1
            else:
                self._active_family     = selected_idx
                self._persistence_ticks = 1

        # --- Tactic selection ---
        logits = self._tactic_logits(selected_idx, obs_arr)
        action_mask   = self._available_mask(selected_idx)
        action = int(np.argmax(np.where(action_mask, logits, -1e9)))

        # Family-level accessibility mask: 7-bit, True = A(f,t) > 0
        # This is what topology_suite uses to detect collapse/recovery
        family_mask = [accessibility.get(fp.name, 0.0) > 0.0 for fp in self._families]

        # Encode family accessibility into available_mask positions 0-6
        # Positions 7-19 retain per-action mask for the active family
        combined_mask = family_mask + action_mask[7:]

        # --- Explanation trace (§M2.6) ---
        if self.cfg.trace_enabled and forced_state is None:
            entry = TraceEntry(
                tick=tick,
                active_family=selected_name,
                policy_scores=dict(raw_scores),
                accessibility=dict(accessibility),
                accessible_families=accessible_families,
                selected_family=selected_name,
                precedence_tag=precedence.value,
                switch_occurred=switch_occurred,
                switch_cost_paid=switch_cost_paid,
                rd=rd,
                urgency=urgency,
            )
            if len(self._explanation_trace) < self.cfg.max_trace_length:
                self._explanation_trace.append(entry)

        tactic_class = f"tactic_{selected_name.lower()}"

        return _M2StepOutput(
            action_logits=logits.tolist(),
            available_mask=combined_mask,
            active_state=selected_idx,
            action_taken=action,
            tactic_class=tactic_class,
            _precedence_tag=precedence,
            _dominant_module=dominant,
        )

    def reset(self, seed: int = 0) -> None:
        self._rng               = np.random.default_rng(seed)
        self._active_family     = 0
        self._persistence_ticks = 0
        self._last_switch_tick  = 0
        self._explanation_trace = []

    def explanation_trace(self) -> List[TraceEntry]:
        return list(self._explanation_trace)

    def accessible_families_at(self, obs: List[float], rd: float) -> List[str]:
        """Utility for battery: which families are accessible at this rd value?"""
        obs_arr = np.array(obs, dtype=np.float32)
        _, _, accessible = self.score_families(obs_arr, rd)
        return accessible

    def param_count(self) -> int:
        return self._W.size

    def _default_family_params(self) -> List[FamilyParams]:
        """Reference params from yaml_validator.generate_reference_config()."""
        return [
            FamilyParams("DEFEND",    rd_threshold=0.80, rd_sensitivity=0.20, urgency_sensitivity=-0.50),
            FamilyParams("WITHDRAW",  rd_threshold=0.68, rd_sensitivity=0.30, urgency_sensitivity=-0.10),
            FamilyParams("REPAIR",    rd_threshold=0.25, rd_sensitivity=0.80, urgency_sensitivity=0.20),
            FamilyParams("EXPLORE",   rd_threshold=0.30, rd_sensitivity=0.60, urgency_sensitivity=0.30),
            FamilyParams("DOMINATE",  rd_threshold=0.55, rd_sensitivity=0.40, urgency_sensitivity=-0.30),
            FamilyParams("SEEK_HELP", rd_threshold=0.35, rd_sensitivity=0.70, urgency_sensitivity=0.40),
            FamilyParams("DECEIVE",   rd_threshold=0.45, rd_sensitivity=0.50, urgency_sensitivity=0.10),
        ]


# ──────────────────────────────────────────────────────────────
# Factory: build wired M2AgentWrapper with real policy layer
# ──────────────────────────────────────────────────────────────

def build_m2_agent(
    num_actions: int   = 128,
    yaml_config: dict  = None,
    seed:        int   = 42,
    agent_id:    str   = "m2_agent_0",
    regime:      str   = "PEACETIME",
    telemetry          = None,
) -> "M2AgentWrapper":
    """
    Build a fully wired M2AgentWrapper with M2MinimalPolicy injected.
    This is the production-ready agent for Regime 1.

    Args:
        yaml_config: dict from yaml_validator.generate_reference_config() or real YAML
        num_actions:  action space size
    """
    from agent_wrapper import M2AgentWrapper

    if yaml_config is None:
        from yaml_validator import generate_reference_config
        yaml_config = generate_reference_config()

    policy_cfg = M2PolicyConfig.from_yaml_config(yaml_config, num_actions=num_actions)
    policy     = M2MinimalPolicy(policy_cfg, seed=seed)

    wrapper = M2AgentWrapper(
        num_actions=num_actions,
        policy_layer=policy,
        telemetry=telemetry,
        agent_id=agent_id,
        regime=regime,
        seed=seed,
    )
    return wrapper


def build_lsm_agent(
    num_actions: int = 128,
    obs_dim:     int = 64,
    seed:        int = 42,
    agent_id:    str = "lsm_agent_0",
    telemetry        = None,
) -> "LSMAgentWrapper":
    """Build a fully wired LSM agent for Regime 1."""
    from agent_wrapper import LSMAgentWrapper
    from lsm_model import LSMConfig, LSMRuntime, LSMModel

    cfg     = LSMConfig(num_actions=num_actions, obs_dim=obs_dim, seed=seed)
    runtime = LSMRuntime(cfg, )
    model   = LSMModel(runtime)

    return LSMAgentWrapper(
        num_actions=num_actions,
        lsm_model=model,
        telemetry=telemetry,
        agent_id=agent_id,
        seed=seed,
    )
