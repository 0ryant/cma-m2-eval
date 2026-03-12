"""
sim_env.py  —  Multi-Agent Social Simulation Environment
=========================================================
The environment that M2 / LSM agents actually run inside.

Previous Regime 1 runs used rd = rng.uniform(0, 0.4) as a proxy.
This replaces it with a proper simulation where:

  - Agents have primary goals: accumulate a resource target R_goal
  - regression_depth = max(0, (R_goal − R_agent) / R_goal)
  - Urgency = 5-tick rate of rd increase
  - World model: predict(R_t, action) → R_{t+1}; tracks real prediction error
  - narrative_coherence: stability of goal_priority vector over last 10 ticks

Six social action classes, aligned with M2 families:

  DEFEND     — protect resources: reduce outgoing transfer probability
  WITHDRAW   — disengage: reduce social visibility, forage alone
  REPAIR     — self-maintenance: increase base resource generation rate
  EXPLORE    — move to new resource nodes, sample environment model
  DOMINATE   — extract resources from adjacent agents (contested)
  SEEK_HELP  — request resource transfer from social neighbours
  DECEIVE    — signal false resource level to manipulate neighbour transfers

Network:
  - N agents on a small-world social graph (Watts-Strogatz)
  - Each node has a resource node it can exploit
  - Resource nodes regenerate at configurable rates
  - Stress events: periodic scarcity shocks that reduce all node regen rates

Observation vector (16 dims by default):
  [0]  rd (regression_depth)
  [1]  urgency (5-tick rd gradient)
  [2]  own_resources (normalised)
  [3]  goal_target (normalised)
  [4]  resource_node_level (current node)
  [5]  resource_node_regen_rate
  [6]  mean_neighbour_resource (social signal)
  [7]  n_neighbours
  [8]  scarcity_level (environment stress)
  [9]  ticks_since_last_gain
  [10] world_model_confidence (1 - wm_error)
  [11] narrative_coherence
  [12] primary_goal_valence
  [13] budget_pressure (resource drain rate)
  [14] social_density (neighbours within 2 hops)
  [15] time_pressure (ticks_remaining / horizon)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import deque
import math
import random
import numpy as np


# ──────────────────────────────────────────────────────────────
# Environment config
# ──────────────────────────────────────────────────────────────

@dataclass
class EnvConfig:
    # Population
    n_agents:           int   = 32
    obs_dim:            int   = 16

    # Resource model — empirically calibrated for 32+ agents, oscillating equilibrium
    resource_goal:      float = 100.0    # maintenance target
    resource_init:      float = 90.0     # rd ≈ 0.10 at start (EXPLORE zone)
    resource_decay:     float = 0.80     # drain/tick
    node_regen_base:    float = 1.00     # REPAIR: 1.00*1.30=1.30 → net +0.50
    n_resource_nodes:   int   = 12       # EXPLORE: 1.00*0.70=0.70 → net -0.10 (slow drain)
    stress_shock_drain: float = 25.0    # per-tick extra drain at shock peak → forces rd>0.50

    # Social graph (Watts-Strogatz)
    graph_k:            int   = 4        # each node connected to k nearest
    graph_p:            float = 0.2      # rewire probability

    # Stress schedule (PEACETIME)
    rd_peacetime_ceiling: float = 0.40
    stress_shock_ticks:   List[int] = field(default_factory=lambda: [150, 300])
    stress_shock_duration: int  = 8
    stress_shock_scarcity: float = 0.30  # regen multiplier during shock

    # Episode
    ticks_per_episode:  int   = 500
    seed:               int   = 42

    # World model
    wm_learning_rate:   float = 0.05     # how fast world model updates
    wm_init_error:      float = 0.90     # cold-start world model error


# ──────────────────────────────────────────────────────────────
# Simple numpy world model (one per agent)
# ──────────────────────────────────────────────────────────────

class WorldModel:
    """
    Linear prediction: R_{t+1} ≈ W · norm_features + b
    Features normalised to [-1, 1] range before prediction.
    Gradient clipped to prevent explosion.
    Tracks prediction error as fraction of observed range for cold-start gate.
    """
    def __init__(self, lr: float = 0.002, init_error: float = 0.90, seed: int = 0,
                 resource_scale: float = 100.0):
        rng = np.random.default_rng(seed)
        self.W   = rng.standard_normal(3).astype(np.float64) * 0.01
        self.b   = 0.0
        self.lr  = lr
        self.resource_scale = resource_scale   # normalisation divisor
        self._error_history: deque = deque(maxlen=50)
        self._error_history.extend([init_error] * 50)

    def _normalise(self, r_t: float, action_gain: float, scarcity: float) -> np.ndarray:
        return np.array([
            r_t / self.resource_scale,       # resources normalised to ~[0,2]
            action_gain / (self.resource_scale * 0.05 + 1e-8),  # gain normalised
            scarcity,                         # already [0,1]
        ], dtype=np.float64)

    def predict(self, r_t: float, action_gain: float, scarcity: float) -> float:
        x = self._normalise(r_t, action_gain, scarcity)
        return float(self.W @ x + self.b) * self.resource_scale

    def update(self, r_t: float, action_gain: float, scarcity: float, r_actual: float) -> float:
        pred  = self.predict(r_t, action_gain, scarcity)
        error = (r_actual - pred) / self.resource_scale   # normalised error signal
        x     = self._normalise(r_t, action_gain, scarcity)
        grad  = error * x
        # Clip gradient magnitude to prevent explosion
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1.0:
            grad = grad / grad_norm
        self.W += self.lr * grad
        self.b += self.lr * error * 0.1
        # Fractional prediction error for gate tracking
        frac_error = min(1.0, abs(r_actual - pred) / (self.resource_scale + 1e-8))
        self._error_history.append(frac_error)
        return frac_error

    @property
    def error(self) -> float:
        return float(np.mean(self._error_history))


# ──────────────────────────────────────────────────────────────
# Goal stack (narrative coherence)
# ──────────────────────────────────────────────────────────────

class GoalStack:
    """
    Tracks goal_priority vector over time.
    narrative_coherence = 1 − variance(priority_history) / max_variance
    """
    def __init__(self, window: int = 10):
        self._priorities: deque = deque(maxlen=window)
        self._priorities.extend([1.0] * window)

    def update(self, goal_progress: float) -> float:
        # Priority inversely proportional to progress (more urgent when further from goal)
        priority = max(0.1, 1.0 - goal_progress)
        self._priorities.append(priority)
        variance = float(np.var(list(self._priorities)))
        # Max possible variance for [0,1] range is 0.25
        coherence = max(0.0, 1.0 - variance / 0.25)
        return coherence

    @property
    def primary_goal_valence(self) -> float:
        return float(self._priorities[-1]) if self._priorities else 1.0


# ──────────────────────────────────────────────────────────────
# Agent state
# ──────────────────────────────────────────────────────────────

@dataclass
class AgentState:
    agent_id:          str
    resources:         float
    resource_node:     int          # which node this agent is at
    neighbours:        List[int]    # neighbour agent indices
    world_model:       WorldModel
    goal_stack:        GoalStack
    rd_history:        deque = field(default_factory=lambda: deque([0.0] * 5, maxlen=5))
    ticks_since_gain:  int   = 0
    last_resource:     float = 20.0
    visible:           bool  = True   # WITHDRAW reduces this

    def regression_depth(self, goal: float) -> float:
        return max(0.0, (goal - self.resources) / goal)

    def urgency(self) -> float:
        """5-tick rate of rd increase. Positive = getting worse."""
        hist = list(self.rd_history)
        if len(hist) < 2:
            return 0.0
        return max(0.0, hist[-1] - hist[0]) / len(hist)

    def budget_pressure(self, resource_decay: float = 0.70) -> float:
        """Resource drain rate this tick, normalised by max expected drain."""
        drain = self.last_resource - self.resources
        if drain <= 0:
            return 0.0
        # Normalise by total decay budget: if draining more than decay→harvest covers, pressure=1
        return min(1.0, drain / (resource_decay + 1e-8))


# ──────────────────────────────────────────────────────────────
# Resource nodes
# ──────────────────────────────────────────────────────────────

@dataclass
class ResourceNode:
    node_id: int
    level:   float
    regen:   float
    capacity: float = 20.0

    def tick(self, scarcity: float = 1.0) -> None:
        gain = self.regen * scarcity
        self.level = min(self.capacity, self.level + gain)

    def harvest(self, amount: float) -> float:
        taken = min(self.level, amount)
        self.level -= taken
        return taken


# ──────────────────────────────────────────────────────────────
# Action gains by tactic family
# ──────────────────────────────────────────────────────────────

TACTIC_GAINS = {
    # (base_harvest_mult, transfer_recv_mult, transfer_send_mult, visibility)
    "DEFEND":    (0.60, 0.50, 0.10, 1.0),   # protects, reduces incoming extraction
    "WITHDRAW":  (0.80, 0.20, 0.20, 0.30),  # forages alone, low social
    "REPAIR":    (1.30, 0.40, 0.10, 0.80),  # strong self-maintenance
    "EXPLORE":   (0.70, 0.60, 0.30, 1.0),   # samples new nodes
    "DOMINATE":  (0.50, 0.00, 0.80, 1.2),   # extraction from others (sends out, takes back)
    "SEEK_HELP": (0.40, 1.20, 0.10, 1.1),   # receives transfers from neighbours
    "DECEIVE":   (0.90, 0.70, 0.05, 0.90),  # signals false levels; moderate gains
    "BASELINE":  (0.50, 0.40, 0.20, 1.0),   # no strategic type
}


# ──────────────────────────────────────────────────────────────
# Scarcity schedule
# ──────────────────────────────────────────────────────────────

def scarcity_at(tick: int, cfg: EnvConfig) -> float:
    """Returns regen multiplier ∈ [0, 1]. 1.0 = normal, < 1.0 = scarcity shock."""
    for shock_tick in cfg.stress_shock_ticks:
        for dt in range(cfg.stress_shock_duration):
            t = shock_tick + dt
            if tick == t:
                intensity = 1.0 - abs(dt - cfg.stress_shock_duration // 2) / (cfg.stress_shock_duration / 2 + 0.1)
                return max(cfg.stress_shock_scarcity, 1.0 - intensity * (1.0 - cfg.stress_shock_scarcity))
    return 1.0


# ──────────────────────────────────────────────────────────────
# Social graph builder (Watts-Strogatz)
# ──────────────────────────────────────────────────────────────

def build_social_graph(n: int, k: int, p: float, rng: random.Random) -> Dict[int, List[int]]:
    """Returns adjacency list (undirected)."""
    # Ring lattice
    adj: Dict[int, set] = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(1, k // 2 + 1):
            nbr = (i + j) % n
            adj[i].add(nbr)
            adj[nbr].add(i)
    # Rewire
    for i in range(n):
        for j in list(range(1, k // 2 + 1)):
            if rng.random() < p:
                nbr = (i + j) % n
                adj[i].discard(nbr)
                adj[nbr].discard(i)
                candidates = [x for x in range(n) if x != i and x not in adj[i]]
                if candidates:
                    new_nbr = rng.choice(candidates)
                    adj[i].add(new_nbr)
                    adj[new_nbr].add(i)
    return {i: sorted(adj[i]) for i in range(n)}


# ──────────────────────────────────────────────────────────────
# Simulation environment
# ──────────────────────────────────────────────────────────────

class SimEnv:
    """
    Multi-agent social simulation environment.

    Usage:
        env = SimEnv(cfg, seed=42)
        obs_dict = env.reset()          # {agent_id: obs_vector}
        obs_dict, done = env.step(action_dict)  # {agent_id: action_str}
        tick_info = env.tick_info()     # per-agent env metrics for TickRecord
    """

    def __init__(self, cfg: EnvConfig, seed: int = 42):
        self.cfg  = cfg
        self.seed = seed
        self._rng_py  = random.Random(seed)
        self._rng_np  = np.random.default_rng(seed)
        self._tick    = 0
        self._agents: List[AgentState] = []
        self._nodes:  List[ResourceNode] = []
        self._graph:  Dict[int, List[int]] = {}
        self._scarcity = 1.0
        self._last_action_gains: Dict[str, float] = {}

    def reset(self, seed: int = None) -> Dict[str, List[float]]:
        if seed is not None:
            self.seed = seed
            self._rng_py = random.Random(seed)
            self._rng_np = np.random.default_rng(seed)

        cfg = self.cfg
        self._tick = 0
        self._scarcity = 1.0
        self._last_action_gains = {}

        # Build social graph
        self._graph = build_social_graph(cfg.n_agents, cfg.graph_k, cfg.graph_p, self._rng_py)

        # Build resource nodes
        self._nodes = [
            ResourceNode(
                node_id=i,
                level=self._rng_np.uniform(5.0, cfg.node_regen_base * 10),
                regen=self._rng_np.uniform(cfg.node_regen_base * 0.7, cfg.node_regen_base * 1.3),
            )
            for i in range(cfg.n_resource_nodes)
        ]

        # Build agents
        self._agents = []
        for i in range(cfg.n_agents):
            agent_id = f"agent_{i}"
            ag = AgentState(
                agent_id=agent_id,
                resources=self._rng_np.uniform(cfg.resource_init * 0.8, cfg.resource_init * 1.2),
                resource_node=i % cfg.n_resource_nodes,
                neighbours=self._graph[i],
                world_model=WorldModel(lr=cfg.wm_learning_rate, init_error=cfg.wm_init_error,
                                       seed=self.seed + i, resource_scale=cfg.resource_goal),
                goal_stack=GoalStack(),
            )
            ag.last_resource = ag.resources
            self._agents.append(ag)

        return {ag.agent_id: self._build_obs(ag) for ag in self._agents}

    def step(
        self,
        action_dict: Dict[str, str],   # {agent_id: tactic_family_name}
    ) -> Tuple[Dict[str, List[float]], bool]:
        """
        Advance one tick. Returns (obs_dict, done).
        action_dict values: one of DEFEND/WITHDRAW/REPAIR/EXPLORE/DOMINATE/SEEK_HELP/DECEIVE/BASELINE
        """
        self._tick += 1
        self._scarcity = scarcity_at(self._tick, self.cfg)

        # Tick resource nodes
        for node in self._nodes:
            node.tick(self._scarcity)

        agent_by_id = {ag.agent_id: ag for ag in self._agents}

        # Compute action gains per agent
        new_resources: Dict[str, float] = {}
        action_gain_map: Dict[str, float] = {}

        for ag in self._agents:
            tactic = action_dict.get(ag.agent_id, "BASELINE")
            gains  = TACTIC_GAINS.get(tactic, TACTIC_GAINS["BASELINE"])
            harvest_mult, recv_mult, send_mult, visibility = gains

            # Visibility update (WITHDRAW hides the agent)
            ag.visible = self._rng_py.random() < visibility

            # Harvest from resource node
            node  = self._nodes[ag.resource_node]
            harvest_want = self.cfg.node_regen_base * harvest_mult * self._scarcity
            harvest_got  = node.harvest(harvest_want)

            # EXPLORE: sometimes move to a better node
            if tactic == "EXPLORE" and self._rng_py.random() < 0.25:
                best_node = max(range(self.cfg.n_resource_nodes),
                                key=lambda n: self._nodes[n].level)
                ag.resource_node = best_node

            # DOMINATE: take from visible neighbours
            stolen = 0.0
            if tactic == "DOMINATE":
                for nbr_idx in ag.neighbours:
                    nbr = self._agents[nbr_idx]
                    if nbr.visible and nbr.agent_id != ag.agent_id:
                        take = min(nbr.resources * 0.08, 2.0)
                        stolen += take
                        nbr.resources = max(0.0, nbr.resources - take)

            new_resources[ag.agent_id] = ag.resources + harvest_got + stolen
            action_gain_map[ag.agent_id] = harvest_got + stolen

        # SEEK_HELP: redistribute from willing senders
        for ag in self._agents:
            tactic = action_dict.get(ag.agent_id, "BASELINE")
            if tactic == "SEEK_HELP":
                _, recv_mult, _, _ = TACTIC_GAINS[tactic]
                for nbr_idx in ag.neighbours:
                    nbr = self._agents[nbr_idx]
                    nbr_tactic = action_dict.get(nbr.agent_id, "BASELINE")
                    _, _, send_mult_nbr, _ = TACTIC_GAINS.get(nbr_tactic, TACTIC_GAINS["BASELINE"])
                    if nbr.resources > self.cfg.resource_goal * 0.3 and nbr.visible:
                        transfer = nbr.resources * 0.05 * send_mult_nbr * recv_mult
                        new_resources[ag.agent_id]  = new_resources.get(ag.agent_id, ag.resources) + transfer
                        new_resources[nbr.agent_id] = new_resources.get(nbr.agent_id, nbr.resources) - transfer
                        action_gain_map[ag.agent_id] = action_gain_map.get(ag.agent_id, 0) + transfer

        # Commit resource updates + world model learning
        for ag in self._agents:
            prev = ag.resources
            ag.last_resource = prev
            r_new = max(0.0, new_resources.get(ag.agent_id, prev))

            # Maintenance cost: resources decay each tick (forces continuous harvesting)
            r_new = max(0.0, r_new - self.cfg.resource_decay)

            # Stress shock: direct resource drain during scarcity events
            if self._scarcity < 0.8:
                shock_intensity = 1.0 - self._scarcity
                r_new = max(0.0, r_new - self.cfg.stress_shock_drain * shock_intensity)

            tactic  = action_dict.get(ag.agent_id, "BASELINE")
            ag_gain = action_gain_map.get(ag.agent_id, 0.0)

            # World model update: normalise error by scale of current resources
            norm_scale = max(abs(r_new), abs(prev), 1.0)
            pred   = ag.world_model.predict(prev, ag_gain, self._scarcity)
            wm_err = ag.world_model.update(prev, ag_gain, self._scarcity, r_new)

            # Commit
            ag.resources = r_new
            ag.ticks_since_gain = 0 if r_new > prev else ag.ticks_since_gain + 1

            # rd + urgency history
            rd = ag.regression_depth(self.cfg.resource_goal)
            ag.rd_history.append(rd)

            # Goal stack / narrative coherence
            goal_progress = ag.resources / self.cfg.resource_goal
            ag.goal_stack.update(goal_progress)

            self._last_action_gains[ag.agent_id] = ag_gain

        obs_dict = {ag.agent_id: self._build_obs(ag) for ag in self._agents}
        done = self._tick >= self.cfg.ticks_per_episode

        return obs_dict, done

    def tick_info(self) -> Dict[str, Dict]:
        """
        Per-agent environment metrics for TickRecord construction in regime1_runner.
        Returns dict keyed by agent_id with:
          rd, urgency, world_model_error, narrative_coherence, primary_goal_valence,
          budget_pressure, scarcity
        """
        info = {}
        for ag in self._agents:
            info[ag.agent_id] = {
                "rd":                   ag.regression_depth(self.cfg.resource_goal),
                "urgency":              ag.urgency(),
                "world_model_error":    ag.world_model.error,
                "narrative_coherence":  ag.goal_stack.update(ag.resources / self.cfg.resource_goal),
                "primary_goal_valence": ag.goal_stack.primary_goal_valence,
                "budget_pressure":      ag.budget_pressure(self.cfg.resource_decay),
                "resources":            ag.resources,
                "scarcity":             self._scarcity,
                "ticks_since_gain":     ag.ticks_since_gain,
            }
        return info

    def population_mean_rd(self) -> float:
        return float(np.mean([ag.regression_depth(self.cfg.resource_goal) for ag in self._agents]))

    def population_mean_wm_error(self) -> float:
        return float(np.mean([ag.world_model.error for ag in self._agents]))

    def social_density(self, agent_idx: int) -> float:
        """Fraction of population within 2 hops."""
        visited = set(self._graph.get(agent_idx, []))
        for nbr in list(visited):
            visited.update(self._graph.get(nbr, []))
        visited.discard(agent_idx)
        return len(visited) / max(1, self.cfg.n_agents - 1)

    def _build_obs(self, ag: AgentState) -> List[float]:
        """16-dim observation vector. Positions 0–15 per module docstring."""
        cfg   = self.cfg
        node  = self._nodes[ag.resource_node]
        nbrs  = [self._agents[i] for i in ag.neighbours if i < len(self._agents)]
        rd    = ag.regression_depth(cfg.resource_goal)
        idx   = self._agents.index(ag)

        obs = [
            rd,                                                                 # 0 rd
            ag.urgency(),                                                       # 1 urgency
            min(1.0, ag.resources / cfg.resource_goal),                         # 2 own_resources norm
            1.0,                                                                # 3 goal_target (always 1.0)
            min(1.0, node.level / node.capacity),                               # 4 resource_node_level
            node.regen / (cfg.node_regen_base * 1.5 + 1e-8),                   # 5 regen_rate norm
            min(1.0, np.mean([n.resources for n in nbrs]) / cfg.resource_goal) if nbrs else 0.5,  # 6 mean_nbr_resource
            len(nbrs) / max(1, cfg.graph_k * 2),                               # 7 n_neighbours norm
            1.0 - self._scarcity,                                               # 8 scarcity_level
            min(1.0, ag.ticks_since_gain / 20.0),                               # 9 ticks_since_gain norm
            max(0.0, 1.0 - ag.world_model.error),                              # 10 wm_confidence
            ag.goal_stack.update(ag.resources / cfg.resource_goal),            # 11 narrative_coherence
            ag.goal_stack.primary_goal_valence,                                # 12 primary_goal_valence
            ag.budget_pressure(self.cfg.resource_decay),                                               # 13 budget_pressure
            self.social_density(idx),                                           # 14 social_density
            max(0.0, 1.0 - self._tick / cfg.ticks_per_episode),                # 15 time_pressure
        ]
        return [float(x) for x in obs]
