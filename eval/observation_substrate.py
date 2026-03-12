"""
observation_substrate.py
========================
Extracts the four-field observation vector that feeds the Social Signal Suite's
ObserverModel. Bridge between CMA runtime telemetry and AgentWrapper.context_vector().

Four fields (Gemini R2, §M2.11.4):
  1. current_persistence_count   — ticks in current active_state, normalised
  2. primary_goal_valence        — goal_stack[0] progress signal ∈ [-1, +1]
  3. rd_delta                    — stress trajectory ∈ [-1, +1]
  4. action_history_embedding    — last-3 action embeddings, mean-pooled [D]

Design constraint (§M2.11.4):
  Both M2 and LSM wrappers call identical update_substrate_buffer() and
  extract_observation_vector() with identical signatures. Context parity is
  a hard requirement — the social signal test must not be contaminated by
  different substrate richness.

Dimension layout:
  Field 1: [1]    persistence_count / persistence_clip ∈ [0, 1]
  Field 2: [1]    primary_goal_valence ∈ [-1, +1]
  Field 3: [1]    rd_delta ∈ [-1, +1]
  Field 4: [D]    action_history mean embedding

Total context_dim = 3 + D

numpy-only. No torch dependency.
"""

from __future__ import annotations
from dataclasses import dataclass, field as dc_field
from typing import List, Dict, Any, Optional
from collections import deque

import numpy as np


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

@dataclass
class ObsSubstrateConfig:
    tactic_embedding_dim: int   = 32     # D
    action_history_len:   int   = 3      # last N actions to mean-pool
    persistence_clip:     int   = 30     # normalisation ceiling
    rd_delta_clip:        float = 0.20   # clip |Δrd| to this before normalising
    goal_valence_clip:    float = 1.0    # clip to [-1, +1]


# ──────────────────────────────────────────────────────────────
# Runtime buffer
# ──────────────────────────────────────────────────────────────

@dataclass
class ObsSubstrateBuffer:
    config:               ObsSubstrateConfig
    persistence_count:    int   = 0
    last_active_state:    int   = -1
    last_rd:              float = 0.0
    primary_goal_valence: float = 0.0
    action_history:       object = None   # deque[np.ndarray[D]]

    def __post_init__(self):
        if self.action_history is None:
            self.action_history = deque(maxlen=self.config.action_history_len)


# ──────────────────────────────────────────────────────────────
# Update — call once per tick BEFORE extract
# ──────────────────────────────────────────────────────────────

def update_substrate_buffer(
    buf:                    ObsSubstrateBuffer,
    rd:                     float,
    last_rd:                float,
    obs:                    List[float],
    persistence_tick_count: int,
    primary_goal_valence:   float,
) -> None:
    """
    Signature used by AgentWrapper.context_vector().
    In the full engine: obs contains the action embedding for the tick's chosen action.
    This stub derives a pseudo-embedding from obs — replace with real embedding table.
    """
    buf.persistence_count    = persistence_tick_count
    buf.last_rd              = last_rd
    buf.primary_goal_valence = float(np.clip(primary_goal_valence, -1.0, 1.0))

    # Pseudo embedding from obs (stub — replace with engine embedding table lookup)
    D = buf.config.tactic_embedding_dim
    obs_arr = np.array(
        (obs[:D] if len(obs) >= D else obs + [0.0] * (D - len(obs))),
        dtype=np.float32,
    )
    norm = float(np.linalg.norm(obs_arr))
    emb  = obs_arr / (norm + 1e-8)
    buf.action_history.append(emb)


# ──────────────────────────────────────────────────────────────
# Extract — returns [3 + D] numpy array
# ──────────────────────────────────────────────────────────────

def extract_observation_vector(
    buf:    ObsSubstrateBuffer,
    config: ObsSubstrateConfig,
) -> np.ndarray:
    """Returns context vector shape [3 + D]."""
    cfg = config

    # Field 1: persistence (normalised)
    f1 = np.array(
        [min(buf.persistence_count, cfg.persistence_clip) / cfg.persistence_clip],
        dtype=np.float32,
    )

    # Field 2: goal valence
    f2 = np.array(
        [float(np.clip(buf.primary_goal_valence, -cfg.goal_valence_clip, cfg.goal_valence_clip))],
        dtype=np.float32,
    )

    # Field 3: rd delta (requires current_rd — not stored separately in buf)
    # The agent_wrapper passes last_rd; current_rd not separately tracked here.
    # Wired engine: update_substrate_buffer receives (rd, last_rd) and can compute delta.
    # Stub: zero until engine wires current_rd explicitly.
    f3 = np.array([0.0], dtype=np.float32)

    # Field 4: mean-pooled action history embedding
    D = cfg.tactic_embedding_dim
    if len(buf.action_history) > 0:
        stack = np.stack(list(buf.action_history), axis=0)   # [N, D]
        f4    = stack.mean(axis=0)                            # [D]
        if len(f4) < D:
            f4 = np.pad(f4, (0, D - len(f4)))
        else:
            f4 = f4[:D]
        f4 = f4.astype(np.float32)
    else:
        f4 = np.zeros(D, dtype=np.float32)

    return np.concatenate([f1, f2, f3, f4])  # [3 + D]


# ──────────────────────────────────────────────────────────────
# Refractory floor check
# ──────────────────────────────────────────────────────────────

@dataclass
class RefractoryFloorCheck:
    """
    Validates that hysteresis is architectural, not lag.
    When rd < rd_floor_threshold, M2 must stay in the same active_state
    for at least refractory_minimum ticks. violations > 0 = build bug.
    """
    refractory_minimum:          int   = 15
    rd_floor_threshold:          float = 0.05
    ticks_since_floor_crossed:   int   = 0
    floor_crossed:               bool  = False
    last_active_state_at_floor:  int   = -1
    ticks_stayed_in_floor_state: int   = 0
    violations:                  int   = 0

    def reset(self) -> None:
        self.ticks_since_floor_crossed    = 0
        self.floor_crossed                = False
        self.last_active_state_at_floor   = -1
        self.ticks_stayed_in_floor_state  = 0

    def check(self, rd: float, active_state: int) -> Dict[str, Any]:
        result = {"refractory_active": False, "violation": False, "ticks_elapsed": 0}
        if rd <= self.rd_floor_threshold and not self.floor_crossed:
            self.floor_crossed = True
            self.ticks_since_floor_crossed = 0
            self.last_active_state_at_floor = active_state
        if self.floor_crossed:
            self.ticks_since_floor_crossed += 1
            result["refractory_active"] = True
            result["ticks_elapsed"]     = self.ticks_since_floor_crossed
            if self.ticks_since_floor_crossed <= self.refractory_minimum:
                if active_state != self.last_active_state_at_floor:
                    self.violations += 1
                    result["violation"] = True
                else:
                    self.ticks_stayed_in_floor_state += 1
            else:
                self.reset()
            if rd > self.rd_floor_threshold:
                self.reset()
        return result


# ──────────────────────────────────────────────────────────────
# Substrate class wrappers (used by SocialSignalSuite)
# ──────────────────────────────────────────────────────────────

class M2ObservationSubstrate:
    def __init__(self, config: ObsSubstrateConfig):
        self.config = config
        self.buf    = ObsSubstrateBuffer(config=config)

    def reset(self) -> None:
        self.buf = ObsSubstrateBuffer(config=self.config)

    def update(self, rd: float, last_rd: float, obs: List[float],
               persistence_tick_count: int, primary_goal_valence: float) -> None:
        update_substrate_buffer(self.buf, rd, last_rd, obs, persistence_tick_count, primary_goal_valence)

    def extract(self) -> np.ndarray:
        return extract_observation_vector(self.buf, self.config)


class LSMObservationSubstrate(M2ObservationSubstrate):
    """Identical substrate for LSM — context parity enforced."""
    pass
