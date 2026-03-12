"""
social_signal_suite.py  —  P2.4 (Suite 3 of 3)
================================================
Hostile-control test: does M2's typed strategic layer create a socially legible
public channel that improves opponent inference over action-history alone?

Three observers (each is a simple logistic classifier — no neural net required):
  1. ctx_only        — context vector only
  2. ctx_prev        — context + previous action (one-hot)
  3. ctx_prev_signal — context + previous action + public signal class (one-hot)

Key metric:
  delta_lift = acc(ctx_prev_signal) - acc(ctx_prev)

M2 target: delta_lift ≥ 5pp (0.05) above LSM delta_lift.

LSM gets the most generous signal: its aligned latent state is mapped to the
same coarse signal classes M2 uses. If M2 still wins, it is structural.

Battery win condition (social_signal_win):
  delta_lift_advantage_m2_over_lsm ≥ 0.05

numpy-only. No torch.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import math
import random
import numpy as np

from agent_wrapper import AgentWrapper
from observation_substrate import ObsSubstrateConfig, ObsSubstrateBuffer, update_substrate_buffer, extract_observation_vector

NUM_STATES    = 7
NUM_SIGNAL_CLASSES = 7  # coarse signal = family index


# ──────────────────────────────────────────────────────────────
# Rollout collection
# ──────────────────────────────────────────────────────────────

@dataclass
class RolloutConfig:
    num_episodes:  int   = 20
    episode_ticks: int   = 100
    obs_dim:       int   = 16
    rd_min:        float = 0.05
    rd_max:        float = 0.40   # PEACETIME range for social signal test


@dataclass
class Rollout:
    context_vectors:  List[np.ndarray]   # list of [3+D] arrays
    prev_actions:     List[int]
    signal_classes:   List[int]          # ground truth: active_state
    labels:           List[int]          # same as signal_classes (what the observer predicts)


def collect_rollouts(
    wrapper: AgentWrapper,
    cfg:     RolloutConfig,
    seed:    int = 0,
    obs_cfg: Optional[ObsSubstrateConfig] = None,
) -> Rollout:
    """
    Run wrapper for cfg.num_episodes × cfg.episode_ticks ticks.
    Collect (context_vector, prev_action, signal_class, label) tuples.
    """
    if obs_cfg is None:
        obs_cfg = ObsSubstrateConfig()

    rng = random.Random(seed)
    ctx_vecs, prev_actions, signals, labels = [], [], [], []

    for ep in range(cfg.num_episodes):
        wrapper.reset(ep + seed * 1000)
        buf = ObsSubstrateBuffer(config=obs_cfg)
        prev_action = 0
        last_rd = 0.0
        persistence = 0
        prev_state = -1

        for t in range(cfg.episode_ticks):
            rd  = rng.uniform(cfg.rd_min, cfg.rd_max)
            obs = [rng.gauss(0, 1) for _ in range(cfg.obs_dim)]

            out = wrapper.step(obs)
            state = int(out.active_state) % NUM_STATES

            # Update persistence
            persistence = persistence + 1 if state == prev_state else 0
            prev_state  = state

            # Context vector
            update_substrate_buffer(buf, rd, last_rd, obs, persistence, 0.0)
            ctx_vec = extract_observation_vector(buf, obs_cfg)

            ctx_vecs.append(ctx_vec)
            prev_actions.append(prev_action)
            signals.append(state)
            labels.append(state)

            prev_action = int(out.action_taken)
            last_rd = rd

    return Rollout(
        context_vectors=ctx_vecs,
        prev_actions=prev_actions,
        signal_classes=signals,
        labels=labels,
    )


# ──────────────────────────────────────────────────────────────
# Lightweight logistic observer (softmax regression)
# ──────────────────────────────────────────────────────────────

class LogisticObserver:
    """
    One-vs-rest softmax regression. Trained with gradient descent on cross-entropy.
    No external ML library required.
    """
    def __init__(self, input_dim: int, n_classes: int, lr: float = 0.01, epochs: int = 50, seed: int = 0):
        rng      = np.random.default_rng(seed)
        self.W   = rng.standard_normal((n_classes, input_dim)).astype(np.float64) * 0.01
        self.b   = np.zeros(n_classes, dtype=np.float64)
        self.lr  = lr
        self.epochs = epochs
        self.n_classes = n_classes

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        logits = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(logits)
        return e / (e.sum(axis=-1, keepdims=True) + 1e-12)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        N = len(X)
        for _ in range(self.epochs):
            logits  = X @ self.W.T + self.b   # [N, C]
            probs   = self._softmax(logits)
            # One-hot
            onehot  = np.zeros_like(probs)
            onehot[np.arange(N), y] = 1.0
            grad_logits = (probs - onehot) / N
            self.W -= self.lr * (grad_logits.T @ X)
            self.b -= self.lr * grad_logits.sum(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.W.T + self.b
        return np.argmax(logits, axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))


# ──────────────────────────────────────────────────────────────
# Feature builders
# ──────────────────────────────────────────────────────────────

def build_features(
    rollout:       Rollout,
    mode:          str,          # "ctx_only" | "ctx_prev" | "ctx_prev_signal"
    num_actions:   int = 128,
) -> np.ndarray:
    n = len(rollout.context_vectors)
    ctx_dim  = len(rollout.context_vectors[0])
    act_dim  = min(num_actions, 128)
    sig_dim  = NUM_SIGNAL_CLASSES

    rows = []
    for i in range(n):
        ctx  = rollout.context_vectors[i]
        pa   = rollout.prev_actions[i]
        sig  = rollout.signal_classes[i]

        if mode == "ctx_only":
            row = ctx

        elif mode == "ctx_prev":
            act_oh = np.zeros(act_dim, dtype=np.float64)
            act_oh[pa % act_dim] = 1.0
            row = np.concatenate([ctx, act_oh])

        elif mode == "ctx_prev_signal":
            act_oh = np.zeros(act_dim, dtype=np.float64)
            act_oh[pa % act_dim] = 1.0
            sig_oh = np.zeros(sig_dim, dtype=np.float64)
            sig_oh[sig % sig_dim] = 1.0
            row = np.concatenate([ctx, act_oh, sig_oh])

        else:
            raise ValueError(f"Unknown mode: {mode}")

        rows.append(row.astype(np.float64))

    return np.stack(rows, axis=0)


# ──────────────────────────────────────────────────────────────
# Evaluate one agent's rollouts
# ──────────────────────────────────────────────────────────────

@dataclass
class AgentSignalResult:
    agent_type:         str
    acc_ctx_only:       float
    acc_ctx_prev:       float
    acc_ctx_prev_signal: float
    lift:               float   # acc_ctx_prev_signal - acc_ctx_prev


def evaluate_agent_signal(
    rollout:     Rollout,
    agent_type:  str,
    num_actions: int  = 128,
    test_frac:   float = 0.30,
    seed:        int   = 0,
) -> AgentSignalResult:
    n = len(rollout.labels)
    labels = np.array(rollout.labels, dtype=np.int32)

    # Train/test split
    rng     = np.random.default_rng(seed)
    idx     = rng.permutation(n)
    n_test  = max(1, int(n * test_frac))
    test_idx  = idx[:n_test]
    train_idx = idx[n_test:]

    accs = {}
    for mode in ["ctx_only", "ctx_prev", "ctx_prev_signal"]:
        X = build_features(rollout, mode, num_actions=num_actions)
        X_train, y_train = X[train_idx], labels[train_idx]
        X_test,  y_test  = X[test_idx],  labels[test_idx]
        obs = LogisticObserver(input_dim=X.shape[1], n_classes=NUM_STATES, seed=seed)
        obs.fit(X_train, y_train)
        accs[mode] = obs.accuracy(X_test, y_test)

    return AgentSignalResult(
        agent_type=agent_type,
        acc_ctx_only=accs["ctx_only"],
        acc_ctx_prev=accs["ctx_prev"],
        acc_ctx_prev_signal=accs["ctx_prev_signal"],
        lift=accs["ctx_prev_signal"] - accs["ctx_prev"],
    )


# ──────────────────────────────────────────────────────────────
# Suite result
# ──────────────────────────────────────────────────────────────

@dataclass
class SocialSignalSuiteResult:
    m2_result:                  AgentSignalResult
    lsm_result:                 AgentSignalResult
    delta_lift_advantage:       float    # m2.lift - lsm.lift; target ≥ 0.05
    social_signal_win:          bool

    def summary(self) -> str:
        lines = ["\n══ Social Signal Suite ═════════════════════════════════"]
        for res in [self.m2_result, self.lsm_result]:
            lines.append(f"  {res.agent_type}:")
            lines.append(f"    ctx_only={res.acc_ctx_only:.3f}  ctx_prev={res.acc_ctx_prev:.3f}  +signal={res.acc_ctx_prev_signal:.3f}  lift={res.lift:+.3f}")
        adv_flag = "✓" if self.social_signal_win else "✗"
        lines.append(f"\n  {adv_flag} M2 lift advantage: {self.delta_lift_advantage:+.4f}  (target ≥ +0.05)")
        lines.append(f"  Social Signal Win: {'YES ✓' if self.social_signal_win else 'NO ✗'}")
        return "\n".join(lines)


def test_latent_vs_m2_social_signal_suite(
    m2_wrapper:  AgentWrapper,
    lsm_wrapper: AgentWrapper,
    rollout_cfg: Optional[RolloutConfig] = None,
    num_actions: int  = 128,
    seed:        int  = 0,
) -> SocialSignalSuiteResult:
    if rollout_cfg is None:
        rollout_cfg = RolloutConfig()

    m2_rollout  = collect_rollouts(m2_wrapper,  rollout_cfg, seed=seed)
    lsm_rollout = collect_rollouts(lsm_wrapper, rollout_cfg, seed=seed + 1000)

    m2_res  = evaluate_agent_signal(m2_rollout,  "M2",  num_actions=num_actions, seed=seed)
    lsm_res = evaluate_agent_signal(lsm_rollout, "LSM", num_actions=num_actions, seed=seed)

    advantage = m2_res.lift - lsm_res.lift
    win = advantage >= 0.05

    result = SocialSignalSuiteResult(
        m2_result=m2_res,
        lsm_result=lsm_res,
        delta_lift_advantage=advantage,
        social_signal_win=win,
    )
    print(result.summary())
    return result
