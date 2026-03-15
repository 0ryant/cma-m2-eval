"""
v5_ablation.py — Fast collapse-ordering ablation + bootstrap
Uses topology_suite.run_episode directly (no sim_env).
Also runs the full experiment pipeline for scale-up.
"""
import json, time, random, copy
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from m2_policy import (M2PolicyConfig, M2MinimalPolicy, FamilyParams,
                       build_calibrated_policy, build_m2_agent)
from agent_wrapper import M2AgentWrapper
from topology_suite import (run_episode, CatastropheSchedule, CollapseTrace,
                            spearman_corr, extract_collapse_trace,
                            BIOLOGICAL_COLLAPSE_PRIOR, NUM_STATES)
from run_experiment import ExperimentConfig, run_experiment
from yaml_validator import generate_reference_config

# ──────────────────────────────────────────────────────────────
# 1. ABLATION: topology suite with shuffled thresholds
# ──────────────────────────────────────────────────────────────

def build_ablation_agent(thresholds: Dict[str, float], seed: int = 42) -> M2AgentWrapper:
    """Build M2 agent with custom thresholds for topology suite."""
    policy = build_calibrated_policy(seed=seed, num_actions=20)
    for fp in policy._families:
        if fp.name in thresholds:
            fp.rd_threshold = thresholds[fp.name]
    policy._family_by_name = {f.name: f for f in policy._families}
    wrapper = M2AgentWrapper(num_actions=20, policy_layer=policy,
                             agent_id=f"ablation_{seed}", seed=seed)
    return wrapper

def compute_spearman_for_trace(trace: CollapseTrace) -> float:
    """Same computation as topology_suite.compute_topology_metrics."""
    bio = list(BIOLOGICAL_COLLAPSE_PRIOR)
    order = list(trace.collapse_order)
    bio_rank = [bio.index(k) if k in bio else len(bio) for k in range(NUM_STATES)]
    obs_rank = [order.index(k) if k in order else len(order) for k in range(NUM_STATES)]
    return spearman_corr(bio_rank, obs_rank)

def run_ablation_condition(label: str, thresholds: Dict[str, float], 
                           n_seeds: int = 10) -> Dict:
    sched = CatastropheSchedule()
    rhos = []
    for s in range(n_seeds):
        wrapper = build_ablation_agent(thresholds, seed=s * 42 + 7)
        trace = run_episode(wrapper, seed=s, schedule=sched)
        rho = compute_spearman_for_trace(trace)
        rhos.append(rho)
    
    m = float(np.mean(rhos))
    sd = float(np.std(rhos))
    print(f"  {label:20s}  rho={m:.4f}+/-{sd:.4f}  "
          f"[{min(rhos):.4f},{max(rhos):.4f}]  "
          f"{'PASS' if m >= 0.85 else 'FAIL'}")
    return {
        "label": label, "rho_mean": m, "rho_std": sd,
        "rho_min": float(min(rhos)), "rho_max": float(max(rhos)),
        "rho_values": rhos, "n_seeds": n_seeds,
    }

# Calibrated thresholds from build_calibrated_policy
CAL = {
    "DEFEND": 2.00, "WITHDRAW": 0.65, "REPAIR": 0.18,
    "EXPLORE": 0.24, "DOMINATE": 0.52, "SEEK_HELP": 0.34, "DECEIVE": 0.42,
}

def run_ablation_study():
    """Test biological vs random vs reversed vs uniform orderings."""
    results = []
    NS = 10
    
    # Reference: biological ordering
    results.append(run_ablation_condition("biological", CAL, NS))
    
    # Reversed: swap non-DEFEND thresholds
    non_def = {k: v for k, v in CAL.items() if k != "DEFEND"}
    sk = sorted(non_def, key=non_def.get)  # REPAIR, EXPLORE, SEEK_HELP, DECEIVE, DOMINATE, WITHDRAW
    rv = list(reversed([non_def[k] for k in sk]))
    rev = dict(zip(sk, rv))
    rev["DEFEND"] = 2.0
    results.append(run_ablation_condition("reversed", rev, NS))
    
    # Uniform: all non-DEFEND get same threshold
    m = float(np.mean([v for k, v in CAL.items() if k != "DEFEND"]))
    uni = {k: m for k in CAL}
    uni["DEFEND"] = 2.0
    results.append(run_ablation_condition("uniform", uni, NS))
    
    # 5 random permutations of non-DEFEND thresholds
    for i in range(5):
        rng = random.Random(4000 + i)
        vals = [v for k, v in CAL.items() if k != "DEFEND"]
        rng.shuffle(vals)
        keys = [k for k in CAL if k != "DEFEND"]
        rnd = dict(zip(keys, vals))
        rnd["DEFEND"] = 2.0
        results.append(run_ablation_condition(f"random_{i+1}", rnd, NS))
    
    return results


# ──────────────────────────────────────────────────────────────
# 2. SCALE: full pipeline at 32, 64, 128 agents
# ──────────────────────────────────────────────────────────────

def run_scale_experiments():
    results = []
    for n in [32, 64, 128]:
        print(f"\n  Running scale={n}...")
        yaml_cfg = generate_reference_config()
        cfg = ExperimentConfig(
            num_agents=n, num_seeds=3, ticks_per_episode=900,
            run_flat_baseline=False, run_regime1=True, run_battery=True,
            run_fmb=False, run_ablation=False, run_model_selection=False)
        result = run_experiment(cfg, yaml_cfg)
        
        obs3 = obs5 = soc = None
        for m in result.metrics:
            if "Obs 3" in m.name: obs3 = m.value
            if "Obs 5" in m.name: obs5 = m.value
            if "Social" in m.name or "Obs 4" in m.name: soc = m.value
        
        r = {"n_agents": n, "downgrade": result.downgrade_level,
             "obs3": obs3, "obs5": obs5, "social": soc,
             "battery_win": result.battery_win, "tier_a": result.tier_a_gate_pass}
        results.append(r)
        print(f"  n={n}: obs3={obs3}, tier_a={result.downgrade_level}")
    return results


# ──────────────────────────────────────────────────────────────
# 3. BOOTSTRAP CIs
# ──────────────────────────────────────────────────────────────

def bootstrap_ci(vals, n_boot=10000, ci=0.95):
    a = np.array(vals)
    bs = [float(np.mean(np.random.choice(a, len(a), True))) for _ in range(n_boot)]
    return float(np.mean(a)), float(np.percentile(bs, (1-ci)/2*100)), float(np.percentile(bs, (1+ci)/2*100))

def run_bootstrap():
    """10-seed bootstrap on the topology suite for CIs."""
    sched = CatastropheSchedule()
    rhos = []
    for s in range(10):
        wrapper = build_ablation_agent(CAL, seed=s * 42 + 7)
        trace = run_episode(wrapper, seed=s, schedule=sched)
        rho = compute_spearman_for_trace(trace)
        rhos.append(rho)
    
    rm, rlo, rhi = bootstrap_ci(rhos)
    print(f"  Obs3 Spearman: {rm:.4f} [{rlo:.4f}, {rhi:.4f}] 95% CI")
    return {"mean": rm, "ci_lo": rlo, "ci_hi": rhi, "values": rhos}


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    R = {}
    
    print("="*70 + "\n1. COLLAPSE-ORDERING ABLATION\n" + "="*70)
    R["ablation"] = run_ablation_study()
    
    print("\n" + "="*70 + "\n2. BOOTSTRAP CIs\n" + "="*70)
    R["bootstrap"] = run_bootstrap()
    
    print("\n" + "="*70 + "\n3. SCALE-UP (full pipeline)\n" + "="*70)
    R["scale"] = run_scale_experiments()
    
    elapsed = time.time() - t0
    
    out = Path("/home/claude/v5_results.json")
    def cvt(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o
    with open(out, "w") as f:
        json.dump(R, f, indent=2, default=cvt)
    
    print(f"\n{'='*70}")
    print(f"ALL DONE in {elapsed:.1f}s")
    print(f"{'='*70}\n")
    
    print(f"{'Condition':20s} {'rho':>7s} {'std':>7s} {'min':>7s} {'Pass':>5s}")
    print("-"*48)
    for r in R["ablation"]:
        p = "PASS" if r["rho_mean"] >= 0.85 else "FAIL"
        print(f"{r['label']:20s} {r['rho_mean']:7.4f} {r['rho_std']:7.4f} {r['rho_min']:7.4f} {p:>5s}")
    
    print(f"\nScale:")
    for r in R["scale"]:
        print(f"  n={r['n_agents']:4d}: obs3={r['obs3']}, level={r['downgrade']}")
    
    print(f"\nResults: {out}")
    return R

if __name__ == "__main__":
    main()
