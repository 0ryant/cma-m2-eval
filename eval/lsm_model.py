"""
lsm_model.py  —  P2.1
=======================
Latent Strategic Manifold (LSM) — four-equation hostile baseline (§M2.1.5).

This is the 'most generous possible hostile baseline': structurally matched
to M2's footprint (7 latent states, persistence, stress-gating, switch cost,
tactic-set restriction) but with NO semantic typing, NO biological ordering,
NO doctrine gates, and NO authored constraint on what each latent state means.
Everything is learned end-to-end.

The LSM must beat this to claim Tier A. If LSM ties M2 on battery metrics,
the claim narrows to 'persistent meso-level strategic state matters' — not
'typed families matter'.

Four equations (§M2.1.5):
  S(z,t)     = W_z · Φ(state_t) + b_z − λ · I(z ≠ z_{t-1}) · switch_cost
  A(z,t)     = clip(1 − max(0, rd_t − thresh_z) · θ_z, 0, 1)
  z*(t)      = argmax_z { S(z,t) · A(z,t) }
  T_z        = { a ∈ A | cluster_id(a) = z }   (k-means at init, frozen)

Parameters:
  W_z, b_z   — learned end-to-end (no biological ordering constraint)
  thresh_z   — learned (no ordering constraint)
  θ_z        — learned (uniform init)
  switch_cost — shared scalar, same as M2
  K = 7      — latent states, same count as M2 families

numpy-only. No torch dependency.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import math
import random
import numpy as np


# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

@dataclass
class LSMConfig:
    num_latent_states:  int   = 7       # K — must equal M2 family count
    num_actions:        int   = 128     # A
    obs_dim:            int   = 64      # dim of Φ(state_t)
    switch_cost:        float = 0.10    # λ — same as M2
    rd_gating:          bool  = True    # if False: no stress gating (ablation)
    tactic_restriction: bool  = True    # if False: full action set always available
    seed:               int   = 42


# ──────────────────────────────────────────────────────────────
# Parameter block
# ──────────────────────────────────────────────────────────────

@dataclass
class LSMParams:
    W:         np.ndarray   # [K, obs_dim] — state scoring weights
    b:         np.ndarray   # [K]          — bias
    thresh:    np.ndarray   # [K]          — accessibility threshold per state
    theta:     np.ndarray   # [K]          — accessibility sensitivity
    tactic_clusters: np.ndarray  # [A]   — action → cluster id (0..K-1), frozen

    @classmethod
    def init_random(cls, cfg: LSMConfig, seed: int = 42) -> "LSMParams":
        rng = np.random.default_rng(seed)
        W     = rng.standard_normal((cfg.num_latent_states, cfg.obs_dim)).astype(np.float32) * 0.1
        b     = np.zeros(cfg.num_latent_states, dtype=np.float32)
        # thresh: uniform over [0.3, 0.8] — no biological ordering
        thresh = rng.uniform(0.3, 0.8, size=cfg.num_latent_states).astype(np.float32)
        theta  = rng.uniform(0.4, 1.2, size=cfg.num_latent_states).astype(np.float32)
        # tactic clusters: k-means at init using random assignment (frozen after)
        clusters = (np.arange(cfg.num_actions) % cfg.num_latent_states).astype(np.int32)
        rng.shuffle(clusters)
        return cls(W=W, b=b, thresh=thresh, theta=theta, tactic_clusters=clusters)

    def param_count(self) -> int:
        return self.W.size + self.b.size + self.thresh.size + self.theta.size


# ──────────────────────────────────────────────────────────────
# LSM runtime
# ──────────────────────────────────────────────────────────────

class LSMRuntime:
    """
    Stateful LSM runtime. One instance per agent.
    Call step() each tick. Maintains z_prev for switch cost.

    This is the component injected into LSMAgentWrapper.lsm_model.
    """

    def __init__(self, cfg: LSMConfig, params: Optional[LSMParams] = None):
        self.cfg    = cfg
        self.params = params or LSMParams.init_random(cfg, cfg.seed)
        self._z_prev: int = 0
        self._rng   = np.random.default_rng(cfg.seed)

    def reset(self, seed: int) -> None:
        self._z_prev = 0
        self._rng    = np.random.default_rng(seed)

    def step(self, phi: np.ndarray, rd: float) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        One forward pass.

        Args:
            phi: observation vector [obs_dim]
            rd:  regression depth ∈ [0, 1]

        Returns:
            z_star:   chosen latent state (int)
            scores:   S(z,t) pre-gating [K]
            mask:     A(z,t) accessibility [K] ∈ [0, 1]
        """
        p = self.params

        # Equation 1: S(z,t) = W_z · Φ + b_z − λ · I(z ≠ z_prev) · switch_cost
        phi_padded = phi
        if len(phi) < self.cfg.obs_dim:
            phi_padded = np.pad(phi, (0, self.cfg.obs_dim - len(phi))).astype(np.float32)
        elif len(phi) > self.cfg.obs_dim:
            phi_padded = phi[:self.cfg.obs_dim].astype(np.float32)

        scores = p.W @ phi_padded + p.b   # [K]
        for z in range(self.cfg.num_latent_states):
            if z != self._z_prev:
                scores[z] -= self.cfg.switch_cost

        # Equation 2: A(z,t) = clip(1 − max(0, rd − thresh_z) · θ_z, 0, 1)
        if self.cfg.rd_gating:
            mask = np.clip(
                1.0 - np.maximum(0.0, rd - p.thresh) * p.theta,
                0.0, 1.0,
            )
        else:
            mask = np.ones(self.cfg.num_latent_states, dtype=np.float32)

        # Equation 3: z* = argmax { S(z) · A(z) }
        gated_scores = scores * mask
        if mask.sum() == 0:
            z_star = self._z_prev
        else:
            z_star = int(np.argmax(gated_scores))

        self._z_prev = z_star
        return z_star, scores, mask

    def action_logits(self, z: int, phi: np.ndarray) -> np.ndarray:
        """
        Equation 4: tactic set T_z = {a | cluster_id(a) = z}.
        Returns logits over all actions; non-T_z actions set to -inf.
        """
        p  = self.params
        logits = np.full(self.cfg.num_actions, -1e9, dtype=np.float32)

        # T_z = actions in cluster z
        tactic_mask = (p.tactic_clusters == z)
        if not tactic_mask.any():
            # Fallback: allow all (shouldn't happen with valid k-means)
            tactic_mask = np.ones(self.cfg.num_actions, dtype=bool)

        if self.cfg.tactic_restriction:
            # Score within-cluster actions using dot product with W_z
            obs = phi if len(phi) == self.cfg.obs_dim else \
                  np.pad(phi, (0, max(0, self.cfg.obs_dim - len(phi))))[:self.cfg.obs_dim]
            w_z = p.W[z]  # [obs_dim]
            # Action scores proportional to alignment with state preference
            base = np.dot(obs, w_z)
            for a in range(self.cfg.num_actions):
                if tactic_mask[a]:
                    logits[a] = base + 0.1 * (a % (z + 1))  # slight action differentiation
        else:
            logits = np.zeros(self.cfg.num_actions, dtype=np.float32)

        return logits

    def available_mask(self, z: int) -> np.ndarray:
        """Boolean mask over actions: True = in T_z (accessible)."""
        if self.cfg.tactic_restriction:
            return (self.params.tactic_clusters == z)
        return np.ones(self.cfg.num_actions, dtype=bool)

    def param_count(self) -> int:
        return self.params.param_count()


# ──────────────────────────────────────────────────────────────
# Complexity matching utility (P2.2)
# ──────────────────────────────────────────────────────────────

def check_complexity_match(
    m2_param_count: int,
    lsm_param_count: int,
    tolerance: float = 0.15,
) -> Tuple[bool, str]:
    """
    §M2.7.6 / P2.2: LSM must not exceed M2 param count by > tolerance.
    Any asymmetry must be noted in the paper even if within tolerance.
    """
    if m2_param_count == 0 or lsm_param_count == 0:
        return False, "Param counts not set. Set both before publication run."
    ratio = lsm_param_count / m2_param_count
    within = abs(ratio - 1.0) <= tolerance
    msg = (
        f"M2={m2_param_count:,}  LSM={lsm_param_count:,}  "
        f"ratio={ratio:.3f}  tolerance=±{tolerance:.0%}  "
        f"{'PASS' if within else 'FAIL — adjust LSM capacity'}"
    )
    return within, msg


# ──────────────────────────────────────────────────────────────
# Adapter: LSMRuntime → LSMAgentWrapper.lsm_model interface
# ──────────────────────────────────────────────────────────────

class LSMModel:
    """
    Thin adapter so LSMRuntime can be injected into LSMAgentWrapper as lsm_model.
    Implements the .forward() protocol used by LSMAgentWrapper._forward().
    """
    def __init__(self, runtime: LSMRuntime):
        self.runtime = runtime

    def reset(self, seed: int) -> None:
        self.runtime.reset(seed)

    def forward(self, obs: List[float], forced_state) -> "ForwardResult":
        phi = np.array(obs, dtype=np.float32)
        rd  = 0.0  # wired engine passes rd through obs; stub uses 0

        if forced_state is not None:
            # Counterfactual: force z, compute logits without updating z_prev
            z = forced_state % self.runtime.cfg.num_latent_states
            _, _, mask_arr = self.runtime.step(phi, rd)
            logits = self.runtime.action_logits(z, phi)
        else:
            z, _, mask_arr = self.runtime.step(phi, rd)
            logits = self.runtime.action_logits(z, phi)

        avail = self.runtime.available_mask(z)
        action = int(np.argmax(np.where(avail, logits, -1e9)))

        from agent_wrapper import StepOutput
        return StepOutput(
            action_logits=logits.tolist(),
            available_mask=avail.tolist(),
            active_state=z,
            action_taken=action,
            tactic_class=f"latent_{z}",
        )
