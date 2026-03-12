"""
yaml_validator.py  —  P0.3
============================
Five hard checks that must ALL pass before any Regime run.
Any failure = do not proceed. These are structural pre-conditions,
not calibration preferences.

Run:
    python yaml_validator.py m2_config.yaml
    python yaml_validator.py m2_config.yaml --strict  # exits 1 on any failure

§ references: M2 v1.18 §3, §4.3, §5.2a/b/c, §6.4, CMA v4.1 §36.7
"""

from __future__ import annotations
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

try:
    import yaml
except ImportError:
    yaml = None


# ──────────────────────────────────────────────────────────────
# Required per-family parameter keys
# ──────────────────────────────────────────────────────────────

REQUIRED_FAMILY_KEYS = [
    "rd_threshold",
    "rd_sensitivity",
    "urgency_sensitivity",
    "quality_threshold",
    "quality_sensitivity",
]

ALL_FAMILIES = ["DEFEND", "WITHDRAW", "REPAIR", "EXPLORE", "DOMINATE", "SEEK_HELP", "DECEIVE"]

# Biological collapse order: earlier index = lower rd_threshold (collapses first)
# REPAIR collapses first (lowest threshold), DEFEND collapses last (highest threshold)
BIOLOGICAL_COLLAPSE_ORDER = ["REPAIR", "EXPLORE", "SEEK_HELP", "DECEIVE", "DOMINATE", "WITHDRAW", "DEFEND"]


# ──────────────────────────────────────────────────────────────
# Check implementations
# ──────────────────────────────────────────────────────────────

def check_1_all_families_present(config: Dict) -> Tuple[bool, str]:
    """
    Check 1: All 7 families present with all required parameter keys.
    """
    families = config.get("families", {})
    missing_families = [f for f in ALL_FAMILIES if f not in families]
    if missing_families:
        return False, f"Missing families: {missing_families}"

    missing_params = []
    for family in ALL_FAMILIES:
        fam_cfg = families[family]
        for key in REQUIRED_FAMILY_KEYS:
            if key not in fam_cfg:
                missing_params.append(f"{family}.{key}")
    if missing_params:
        return False, f"Missing family parameters: {missing_params}"

    return True, "All 7 families present with required parameters."


def check_2_biological_collapse_order(config: Dict) -> Tuple[bool, str]:
    """
    Check 2: Biological collapse order satisfied.
    rd_threshold must increase monotonically through:
    REPAIR < EXPLORE < SEEK_HELP < DECEIVE < DOMINATE < WITHDRAW < DEFEND

    This is §M2.4.3 biological constraint. Any parameterisation that
    violates this ordering is biologically implausible and must be rejected.
    """
    families = config.get("families", {})
    thresholds = {}
    for f in BIOLOGICAL_COLLAPSE_ORDER:
        if f not in families:
            return False, f"Family {f} missing — run Check 1 first."
        thresholds[f] = families[f]["rd_threshold"]

    violations = []
    for i in range(len(BIOLOGICAL_COLLAPSE_ORDER) - 1):
        a = BIOLOGICAL_COLLAPSE_ORDER[i]
        b = BIOLOGICAL_COLLAPSE_ORDER[i + 1]
        if thresholds[a] >= thresholds[b]:
            violations.append(
                f"{a}.rd_threshold ({thresholds[a]:.3f}) >= {b}.rd_threshold ({thresholds[b]:.3f}) — "
                f"violation: {a} must collapse before {b}"
            )

    if violations:
        return False, "Biological collapse order violated:\n  " + "\n  ".join(violations)

    order_str = " < ".join(f"{f}({thresholds[f]:.3f})" for f in BIOLOGICAL_COLLAPSE_ORDER)
    return True, f"Collapse order satisfied: {order_str}"


def check_3_seek_help_threshold(config: Dict) -> Tuple[bool, str]:
    """
    Check 3: SEEK_HELP.rd_threshold ∈ [0.30, 0.40]
    Corrected from 0.5 → 0.35 in A.04. Consistency enforced here.
    """
    families = config.get("families", {})
    if "SEEK_HELP" not in families:
        return False, "SEEK_HELP family missing."
    t = families["SEEK_HELP"]["rd_threshold"]
    if not (0.30 <= t <= 0.40):
        return False, (
            f"SEEK_HELP.rd_threshold = {t:.3f} is outside [0.30, 0.40]. "
            f"Corrected value per A.04 is 0.35. Update config."
        )
    return True, f"SEEK_HELP.rd_threshold = {t:.3f} ✓"


def check_4_baseline_thresholds(config: Dict) -> Tuple[bool, str]:
    """
    Check 4: BASELINE controller state thresholds (§M2.5.2a).
    - activation_threshold: max(policy_score) < 0.35 → BASELINE
    - ambiguity_threshold: top-2 within 0.05 AND both < 0.45 → BASELINE
    """
    baseline = config.get("baseline_controller", {})
    errors = []

    activation = baseline.get("activation_threshold")
    if activation is None:
        errors.append("baseline_controller.activation_threshold missing (required: 0.35)")
    elif not (0.32 <= activation <= 0.38):
        errors.append(
            f"baseline_controller.activation_threshold = {activation} outside tolerance [0.32, 0.38]. "
            f"Specified value: 0.35"
        )

    ambiguity_margin = baseline.get("ambiguity_margin")
    if ambiguity_margin is None:
        errors.append("baseline_controller.ambiguity_margin missing (required: 0.05)")
    elif not (0.03 <= ambiguity_margin <= 0.07):
        errors.append(
            f"baseline_controller.ambiguity_margin = {ambiguity_margin} outside tolerance [0.03, 0.07]. "
            f"Specified value: 0.05"
        )

    ambiguity_ceiling = baseline.get("ambiguity_ceiling")
    if ambiguity_ceiling is None:
        errors.append("baseline_controller.ambiguity_ceiling missing (required: 0.45)")
    elif not (0.42 <= ambiguity_ceiling <= 0.48):
        errors.append(
            f"baseline_controller.ambiguity_ceiling = {ambiguity_ceiling} outside tolerance [0.42, 0.48]. "
            f"Specified value: 0.45"
        )

    if errors:
        return False, "BASELINE controller misconfigured:\n  " + "\n  ".join(errors)
    return True, (
        f"BASELINE thresholds valid: activation={activation}, "
        f"ambiguity_margin={ambiguity_margin}, ceiling={ambiguity_ceiling}"
    )


def check_5_deceive_doctrine_gate(config: Dict) -> Tuple[bool, str]:
    """
    Check 5: DECEIVE doctrine gate present and latent_deceive_prior_floor ≥ 0.10.
    DECEIVE is opt-in — doctrine gate prevents accidental deception activation.
    A floor of 0.10 prevents shame spiral runaway (§7.5 stability controls).
    """
    deceive = config.get("deceive_config", {})
    errors = []

    if not deceive.get("doctrine_gate_enabled", False):
        errors.append("deceive_config.doctrine_gate_enabled must be true. DECEIVE is opt-in, not default.")

    floor = deceive.get("latent_deceive_prior_floor")
    if floor is None:
        errors.append("deceive_config.latent_deceive_prior_floor missing (required: ≥ 0.10)")
    elif floor < 0.10:
        errors.append(
            f"deceive_config.latent_deceive_prior_floor = {floor} < 0.10. "
            f"Floor prevents shame–concealment spiral runaway (§7.5). Must be ≥ 0.10."
        )

    shame_ceiling = deceive.get("shame_ceiling")
    if shame_ceiling is None:
        errors.append("deceive_config.shame_ceiling missing (required: 0.85)")
    elif shame_ceiling > 0.90:
        errors.append(
            f"deceive_config.shame_ceiling = {shame_ceiling} > 0.90. "
            f"Ceiling must be ≤ 0.90 to prevent uncapped shame spiral."
        )

    persistence_cap = deceive.get("persistence_hard_cap_ticks")
    if persistence_cap is None:
        errors.append("deceive_config.persistence_hard_cap_ticks missing (required: 25)")
    elif persistence_cap > 30:
        errors.append(
            f"deceive_config.persistence_hard_cap_ticks = {persistence_cap} > 30. "
            f"Hard cap prevents DECEIVE lock-in. Must be ≤ 30."
        )

    if errors:
        return False, "DECEIVE doctrine gate misconfigured:\n  " + "\n  ".join(errors)
    return True, (
        f"DECEIVE gate valid: doctrine_gate=True, "
        f"prior_floor={floor}, shame_ceiling={shame_ceiling}, cap={persistence_cap} ticks"
    )


# ──────────────────────────────────────────────────────────────
# Additional structural checks (warnings, not hard blocks)
# ──────────────────────────────────────────────────────────────

def soft_check_stability_controls(config: Dict) -> List[str]:
    """Warns if §7.5 stability controls are missing — not a hard failure but required before Regime 2."""
    warnings = []
    sc = config.get("stability_controls", {})

    if "DEFEND_refractory_minimum_ticks" not in sc:
        warnings.append("stability_controls.DEFEND_refractory_minimum_ticks missing (recommended: 15)")
    elif sc["DEFEND_refractory_minimum_ticks"] < 10:
        warnings.append(f"DEFEND_refractory_minimum_ticks = {sc['DEFEND_refractory_minimum_ticks']} is < 10 — fear amplification risk")

    if "fear_decay_rate_floor" not in sc:
        warnings.append("stability_controls.fear_decay_rate_floor missing (recommended: 0.05/tick)")

    if "DECEIVE_persistence_hard_cap" not in sc:
        warnings.append("stability_controls.DECEIVE_persistence_hard_cap should mirror deceive_config.persistence_hard_cap_ticks")

    return warnings


def soft_check_persistence(config: Dict) -> List[str]:
    """Warns if persistence minimum is absent or too low — breaks oscillation metric."""
    warnings = []
    pm = config.get("persistence_minimum_ticks")
    if pm is None:
        warnings.append("persistence_minimum_ticks missing (required for volatility gate: recommended 5)")
    elif pm < 3:
        warnings.append(f"persistence_minimum_ticks = {pm} < 3 — oscillation rate will be indistinguishable from LSM")
    return warnings


# ──────────────────────────────────────────────────────────────
# Main validator
# ──────────────────────────────────────────────────────────────

CHECKS = [
    ("Check 1 — All families + parameters present", check_1_all_families_present),
    ("Check 2 — Biological collapse order",          check_2_biological_collapse_order),
    ("Check 3 — SEEK_HELP threshold [0.30, 0.40]",  check_3_seek_help_threshold),
    ("Check 4 — BASELINE controller thresholds",     check_4_baseline_thresholds),
    ("Check 5 — DECEIVE doctrine gate",              check_5_deceive_doctrine_gate),
]


def run_validator(config: Dict, strict: bool = True) -> bool:
    """
    Run all 5 hard checks. Returns True if all pass.
    In strict mode (default), any failure causes early print of all results then returns False.
    """
    print("\n" + "═" * 60)
    print("  M2 YAML Validator — P0.3")
    print("═" * 60)

    all_pass = True
    results = []

    for name, check_fn in CHECKS:
        try:
            passed, msg = check_fn(config)
        except Exception as e:
            passed, msg = False, f"Exception during check: {e}"

        status = "✓ PASS" if passed else "✗ FAIL"
        results.append((name, passed, msg))
        if not passed:
            all_pass = False

    # Print results
    for name, passed, msg in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"\n  {status}  {name}")
        for line in msg.split("\n"):
            colour = "" if passed else "  !!"
            print(f"       {colour} {line}")

    # Soft checks
    warnings = soft_check_stability_controls(config) + soft_check_persistence(config)
    if warnings:
        print("\n  Warnings (not hard failures — wire before Regime 2):")
        for w in warnings:
            print(f"    ⚠  {w}")

    print("\n" + "═" * 60)
    if all_pass:
        print("  RESULT: ALL CHECKS PASSED — safe to proceed")
    else:
        print("  RESULT: VALIDATION FAILED — do NOT run Regime until fixed")
    print("═" * 60 + "\n")

    return all_pass


def load_config(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    text = p.read_text()
    if path.endswith(".json"):
        return json.loads(text)
    if yaml is not None:
        return yaml.safe_load(text)
    # Fallback: try json
    return json.loads(text)


# ──────────────────────────────────────────────────────────────
# Reference config generator
# ──────────────────────────────────────────────────────────────

def generate_reference_config() -> Dict:
    """
    Returns a minimal valid config dict that passes all 5 checks.
    Use as a starting point for your actual simulation config.
    """
    # Biological collapse order: REPAIR < EXPLORE < SEEK_HELP < DECEIVE < DOMINATE < WITHDRAW < DEFEND
    # SEEK_HELP.rd_threshold fixed at 0.35 per A.04/A.93 — EXPLORE must be strictly below 0.35
    return {
        "families": {
            "REPAIR":    {"rd_threshold": 0.25, "rd_sensitivity": 0.80, "urgency_sensitivity": 0.20, "quality_threshold": 0.20, "quality_sensitivity": 0.60},
            "EXPLORE":   {"rd_threshold": 0.30, "rd_sensitivity": 0.60, "urgency_sensitivity": 0.30, "quality_threshold": 0.25, "quality_sensitivity": 0.50},
            "SEEK_HELP": {"rd_threshold": 0.35, "rd_sensitivity": 0.70, "urgency_sensitivity": 0.40, "quality_threshold": 0.28, "quality_sensitivity": 0.55},
            "DECEIVE":   {"rd_threshold": 0.45, "rd_sensitivity": 0.50, "urgency_sensitivity": 0.10, "quality_threshold": 0.38, "quality_sensitivity": 0.40},
            "DOMINATE":  {"rd_threshold": 0.55, "rd_sensitivity": 0.40, "urgency_sensitivity": -0.30, "quality_threshold": 0.48, "quality_sensitivity": 0.30},
            "WITHDRAW":  {"rd_threshold": 0.68, "rd_sensitivity": 0.30, "urgency_sensitivity": -0.10, "quality_threshold": 0.58, "quality_sensitivity": 0.20},
            "DEFEND":    {"rd_threshold": 0.80, "rd_sensitivity": 0.20, "urgency_sensitivity": -0.50, "quality_threshold": 0.70, "quality_sensitivity": 0.10},
        },
        "baseline_controller": {
            "activation_threshold": 0.35,
            "ambiguity_margin":     0.05,
            "ambiguity_ceiling":    0.45,
        },
        "deceive_config": {
            "doctrine_gate_enabled":     True,
            "latent_deceive_prior_floor": 0.10,
            "shame_ceiling":              0.85,
            "persistence_hard_cap_ticks": 25,
        },
        "stability_controls": {
            "DEFEND_refractory_minimum_ticks": 15,
            "fear_decay_rate_floor":           0.05,
            "DECEIVE_persistence_hard_cap":    25,
        },
        "persistence_minimum_ticks": 5,
    }


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M2 YAML config validator — P0.3")
    parser.add_argument("config", nargs="?", help="Path to YAML or JSON config file")
    parser.add_argument("--strict", action="store_true", default=True,
                        help="Exit with code 1 on any failure (default: True)")
    parser.add_argument("--generate", action="store_true",
                        help="Print a reference config that passes all checks")
    args = parser.parse_args()

    if args.generate:
        ref = generate_reference_config()
        if yaml:
            print(yaml.dump(ref, default_flow_style=False))
        else:
            print(json.dumps(ref, indent=2))
        # Self-validate the reference config
        ok = run_validator(ref)
        sys.exit(0 if ok else 1)

    if not args.config:
        print("Usage: python yaml_validator.py <config.yaml>  [--generate]")
        sys.exit(1)

    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    ok = run_validator(config, strict=args.strict)
    sys.exit(0 if ok else 1)
