"""
agent_wrapper.py  —  P0.5
===========================
Abstract base class shared by M2 and LSM wrappers.
Both must implement the same contract — the battery cannot run otherwise.

Three call sites:
  run_lsm_battery.py       — step(), reset()
  counterfactual_suite.py  — step_forced_state()
  social_signal_suite.py   — context_vector()

§ references: M2 v1.18 §11.1, §M2.11.4 (observation substrate), §M2.7.6
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import math

from telemetry import (
    TelemetryEmitter, TickRecord, PrecedenceTagEvent, M2Family, M2Overlay,
    PrecedenceTag, FAMILY_INDEX, resolve_precedence_tag, check_depressive_lock_in,
)
from observation_substrate import (
    ObsSubstrateConfig, ObsSubstrateBuffer, M2ObservationSubstrate,
    LSMObservationSubstrate, update_substrate_buffer, extract_observation_vector,
)


# ──────────────────────────────────────────────────────────────
# Step output — both wrappers return this
# ──────────────────────────────────────────────────────────────

@dataclass
class StepOutput:
    action_logits:   List[float]    # raw logits over action space, len = num_actions
    available_mask:  List[bool]     # True = action available this tick
    active_state:    int            # discrete state index (M2: M2Family enum int; LSM: 0–6)
    action_taken:    int            # argmax(logits * mask)
    tactic_class:    str            # semantic class label


@dataclass
class ContextOutput:
    vector:          List[float]    # [3 + D] observation substrate vector
    dim:             int            # = 3 + D


# ──────────────────────────────────────────────────────────────
# Abstract base — both wrappers must implement
# ──────────────────────────────────────────────────────────────

class AgentWrapper(ABC):
    """
    Shared interface for M2 and LSM agents.
    The battery imports AgentWrapper and calls the interface — never touches internals.
    """

    def __init__(self, num_actions: int, telemetry: Optional[TelemetryEmitter] = None):
        self.num_actions = num_actions
        self.tel = telemetry or TelemetryEmitter()
        self._tick = 0
        self._seed = 0

    @abstractmethod
    def step(self, obs: List[float]) -> StepOutput:
        """
        One tick forward pass.
        obs: flattened agent observation vector
        Returns StepOutput — battery reads action_logits, available_mask, active_state
        """
        ...

    @abstractmethod
    def step_forced_state(self, obs: List[float], state: int) -> StepOutput:
        """
        Counterfactual probe: force agent into state `state`, run step.
        Used by counterfactual_suite.py to measure within/between-state coherence.
        Must NOT update persistent internal state (goal stack, persistence counters, etc.)
        Call semantics: stateless read, not a training step.
        """
        ...

    @abstractmethod
    def context_vector(self, obs: List[float]) -> ContextOutput:
        """
        Return [3 + D] observation substrate vector for social signal suite.
        Must use observation_substrate.py extract_observation_vector() — context parity required.
        Both M2 and LSM must use identical substrate logic.
        """
        ...

    @abstractmethod
    def reset(self, seed: int) -> None:
        """Reset all internal state for a new episode."""
        ...

    @property
    def tick(self) -> int:
        return self._tick


# ──────────────────────────────────────────────────────────────
# M2 Agent Wrapper
# ──────────────────────────────────────────────────────────────

class M2AgentWrapper(AgentWrapper):
    """
    Wraps an M2 policy layer + CMA modules.
    The actual policy_layer, goal_stack, and tactic_set_fn are injected at construction
    — this wrapper is the telemetry + interface layer only.

    Minimal implementation: policy_layer=None runs a deterministic scoring stub
    so the battery scaffolding can be exercised before the sim engine is built.
    """

    def __init__(
        self,
        num_actions: int,
        policy_layer=None,           # M2 policy module (injected from sim engine)
        telemetry: Optional[TelemetryEmitter] = None,
        agent_id: str = "m2_agent_0",
        regime: str = "PEACETIME",
        seed: int = 0,
        obs_config: Optional[ObsSubstrateConfig] = None,
    ):
        super().__init__(num_actions, telemetry)
        self.policy_layer = policy_layer
        self.agent_id = agent_id
        self.regime = regime
        self._seed = seed

        # Observation substrate (§M2.11.4)
        self._obs_cfg = obs_config or ObsSubstrateConfig()
        self._obs_buf = ObsSubstrateBuffer(self._obs_cfg)
        self._substrate = M2ObservationSubstrate(self._obs_cfg)

        # Persistence state
        self._active_family: M2Family = M2Family.BASELINE
        self._persistence_tick_count: int = 0
        self._baseline_ticks: int = 0
        self._mourn_ticks_in_baseline: int = 0
        self._narrative_coherence: float = 1.0
        self._narrative_coherence_baseline: float = 1.0
        self._rd: float = 0.0
        self._last_rd: float = 0.0
        self._world_model_error: float = 0.90  # cold-start value
        self._primary_goal_valence: float = 0.0

    def reset(self, seed: int) -> None:
        self._seed = seed
        self._tick = 0
        self._active_family = M2Family.BASELINE
        self._persistence_tick_count = 0
        self._baseline_ticks = 0
        self._mourn_ticks_in_baseline = 0
        self._narrative_coherence = 1.0
        self._narrative_coherence_baseline = 1.0
        self._rd = 0.0
        self._last_rd = 0.0
        self._world_model_error = 0.90
        self._obs_buf = ObsSubstrateBuffer(self._obs_cfg)

    def step(self, obs: List[float]) -> StepOutput:
        self._tick += 1
        out = self._forward(obs, forced_state=None)

        # Telemetry: PRECEDENCE_TAG every action (§M2.7.6, P0.2)
        self.tel.event(PrecedenceTagEvent(
            tick=self._tick,
            agent_id=self.agent_id,
            event_tag=out._precedence_tag,
            dominant_module=out._dominant_module,
            action_taken=out.action_taken,
            family_active=self._active_family,
            rd_at_action=self._rd,
        ))

        # Telemetry: check depressive lock-in (P0.2)
        check_depressive_lock_in(
            self.tel, self.agent_id, self._tick,
            self._baseline_ticks, self._mourn_ticks_in_baseline,
            self._narrative_coherence, self._narrative_coherence_baseline,
        )

        return out

    def step_forced_state(self, obs: List[float], state: int) -> StepOutput:
        """Counterfactual probe — does NOT update persistent state."""
        saved = (self._active_family, self._persistence_tick_count, self._rd)
        try:
            self._active_family = list(M2Family)[state % 7]
            return self._forward(obs, forced_state=state)
        finally:
            self._active_family, self._persistence_tick_count, self._rd = saved

    def context_vector(self, obs: List[float]) -> ContextOutput:
        update_substrate_buffer(self._obs_buf, self._rd, self._last_rd, obs,
                                self._persistence_tick_count, self._primary_goal_valence)
        vec = extract_observation_vector(self._obs_buf, self._obs_cfg)
        return ContextOutput(vector=list(vec), dim=len(vec))

    # ── internal ──────────────────────────────────────────────

    def _forward(self, obs: List[float], forced_state) -> "_M2StepOutput":
        # If policy layer injected, delegate. Otherwise: scoring stub.
        if self.policy_layer is not None:
            return self.policy_layer.forward(obs, forced_state)

        # Stub: score all families, pick highest accessible
        scores = self._score_families(obs)
        active = self._select_family(scores)
        logits = self._tactic_logits(active, obs)
        mask   = [True] * self.num_actions
        action = max(range(self.num_actions), key=lambda i: logits[i] if mask[i] else -1e9)

        # Precedence resolution
        tag = PrecedenceTag.SCORE_WIN  # stub: always score-driven
        out = _M2StepOutput(
            action_logits=logits,
            available_mask=mask,
            active_state=list(M2Family).index(active),
            action_taken=action,
            tactic_class=f"tactic_{active.value.lower()}",
            _precedence_tag=tag,
            _dominant_module="M2_policy",
        )
        self._active_family = active
        return out

    def _score_families(self, obs: List[float]) -> Dict[M2Family, float]:
        """Stub scoring — replace with real policy_layer scores."""
        import math
        rd = self._rd
        scores = {}
        for f, i in FAMILY_INDEX.items():
            # Simple linear score from obs (placeholder)
            base = sum(obs[j % len(obs)] * (0.1 * (i + 1)) for j in range(4))
            scores[f] = max(0.0, min(1.0, base))
        return scores

    def _select_family(self, scores: Dict[M2Family, float]) -> M2Family:
        """§M2.5.2a BASELINE selection logic."""
        top_score = max(scores.values())
        sorted_scores = sorted(scores.values(), reverse=True)
        top2_within = abs(sorted_scores[0] - sorted_scores[1]) < 0.05
        if top_score < 0.35 or (top2_within and top_score < 0.45):
            self._baseline_ticks += 1
            return M2Family.BASELINE
        self._baseline_ticks = 0
        return max(scores, key=scores.get)

    def _tactic_logits(self, family: M2Family, obs: List[float]) -> List[float]:
        """Stub tactic logits restricted to active family's tactic set."""
        return [float(i % (FAMILY_INDEX.get(family, 0) + 1) == 0) for i in range(self.num_actions)]


class _M2StepOutput(StepOutput):
    """StepOutput extended with internal precedence fields (not exported to battery)."""
    def __init__(self, *, _precedence_tag: PrecedenceTag, _dominant_module: str, **kwargs):
        super().__init__(**kwargs)
        self._precedence_tag = _precedence_tag
        self._dominant_module = _dominant_module


# ──────────────────────────────────────────────────────────────
# LSM Agent Wrapper  (hostile baseline)
# ──────────────────────────────────────────────────────────────

class LSMAgentWrapper(AgentWrapper):
    """
    Wraps the LSM hostile baseline (§M2.1.5 four-equation spec).

    LSM has no semantic family labels, no biological threshold ordering,
    no persistence minimum, no doctrine gate. Everything is learned end-to-end.

    active_state: 0–6 (latent state index — no semantic label)
    """

    def __init__(
        self,
        num_actions: int,
        lsm_model=None,              # LSM model (injected from sim engine)
        telemetry: Optional[TelemetryEmitter] = None,
        agent_id: str = "lsm_agent_0",
        seed: int = 0,
        obs_config: Optional[ObsSubstrateConfig] = None,
    ):
        super().__init__(num_actions, telemetry)
        self.lsm_model = lsm_model
        self.agent_id = agent_id
        self._seed = seed

        # LSM uses identical observation substrate — context parity (§M2.11.4)
        self._obs_cfg = obs_config or ObsSubstrateConfig()
        self._obs_buf = ObsSubstrateBuffer(self._obs_cfg)
        self._substrate = LSMObservationSubstrate(self._obs_cfg)

        self._active_latent_state: int = 0
        self._rd: float = 0.0
        self._last_rd: float = 0.0
        self._persistence_tick_count: int = 0
        self._primary_goal_valence: float = 0.0

    def reset(self, seed: int) -> None:
        self._seed = seed
        self._tick = 0
        self._active_latent_state = 0
        self._rd = 0.0
        self._last_rd = 0.0
        self._persistence_tick_count = 0
        self._obs_buf = ObsSubstrateBuffer(self._obs_cfg)

    def step(self, obs: List[float]) -> StepOutput:
        self._tick += 1
        return self._forward(obs, forced_state=None)

    def step_forced_state(self, obs: List[float], state: int) -> StepOutput:
        saved = (self._active_latent_state, self._rd)
        try:
            self._active_latent_state = state % 7
            return self._forward(obs, forced_state=state)
        finally:
            self._active_latent_state, self._rd = saved

    def context_vector(self, obs: List[float]) -> ContextOutput:
        """LSM uses identical substrate — context parity required (§M2.11.4)."""
        update_substrate_buffer(self._obs_buf, self._rd, self._last_rd, obs,
                                self._persistence_tick_count, self._primary_goal_valence)
        vec = extract_observation_vector(self._obs_buf, self._obs_cfg)
        return ContextOutput(vector=list(vec), dim=len(vec))

    def _forward(self, obs: List[float], forced_state) -> StepOutput:
        if self.lsm_model is not None:
            return self.lsm_model.forward(obs, forced_state)

        # Stub: uniform latent scoring with switch penalty
        K = 7
        switch_cost = 0.10
        scores = [math.sin(obs[i % len(obs)] + i) for i in range(K)]
        if forced_state is None:
            scores[self._active_latent_state] += switch_cost
        state = max(range(K), key=lambda i: scores[i])
        self._active_latent_state = state

        logits = [float(i % (state + 1) == 0) for i in range(self.num_actions)]
        mask   = [True] * self.num_actions
        action = max(range(self.num_actions), key=lambda i: logits[i])

        return StepOutput(
            action_logits=logits,
            available_mask=mask,
            active_state=state,
            action_taken=action,
            tactic_class=f"latent_{state}",
        )


# ──────────────────────────────────────────────────────────────
# Flat U(a) Baseline Wrapper  (P1.1)
# ──────────────────────────────────────────────────────────────

class FlatUAWrapper(AgentWrapper):
    """
    Flat unconstrained utility maximisation.
    No strategic layer, no persistence, no family restriction.
    This is the floor every other agent must beat.

    U(a) = weighted sum over needs × drives × beliefs
    No tactic-set restriction — full action space always available.
    """

    def __init__(self, num_actions: int, telemetry=None, agent_id="flat_ua_0", seed=0):
        super().__init__(num_actions, telemetry)
        self.agent_id = agent_id
        self._seed = seed
        self._rd = 0.0
        self._last_rd = 0.0
        self._persistence_tick_count = 0
        self._primary_goal_valence = 0.0
        self._obs_cfg = ObsSubstrateConfig()
        self._obs_buf = ObsSubstrateBuffer(self._obs_cfg)

    def reset(self, seed: int) -> None:
        self._seed = seed
        self._tick = 0
        self._rd = 0.0
        self._obs_buf = ObsSubstrateBuffer(self._obs_cfg)

    def step(self, obs: List[float]) -> StepOutput:
        self._tick += 1
        # All actions always available — no family restriction
        logits = [sum(obs[j % len(obs)] * ((i + 1) * 0.1) for j in range(3)) for i in range(self.num_actions)]
        mask   = [True] * self.num_actions
        action = max(range(self.num_actions), key=lambda i: logits[i])
        return StepOutput(
            action_logits=logits,
            available_mask=mask,
            active_state=0,            # no state — always 0
            action_taken=action,
            tactic_class="unconstrained",
        )

    def step_forced_state(self, obs: List[float], state: int) -> StepOutput:
        return self.step(obs)  # no state concept — forced state is a no-op

    def context_vector(self, obs: List[float]) -> ContextOutput:
        update_substrate_buffer(self._obs_buf, self._rd, self._last_rd, obs,
                                self._persistence_tick_count, self._primary_goal_valence)
        vec = extract_observation_vector(self._obs_buf, self._obs_cfg)
        return ContextOutput(vector=list(vec), dim=len(vec))
