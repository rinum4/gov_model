
# wp_state_experiment_v1.py
# v1: State manageability sandbox with:
# - management engine from wp_experiment (cases/queues/WIP/escalations/gated actuation + IEKV O/W/E/A)
# - macro state updated via compact closed subgraph inspired by noocracy_bifurcation_skeleton
# - policies act via ScenarioKnobs-like multipliers/shifts (no direct state assignment)

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


# -----------------------------
# helpers
# -----------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def exp_decay_factor(half_life_days: float, dt_days: float = 1.0) -> float:
    if half_life_days <= 0:
        return 0.0
    return math.exp(-math.log(2.0) * dt_days / half_life_days)

def now_day(hour: int, hours_per_day: int) -> int:
    return hour // hours_per_day


# -----------------------------
# Management case
# -----------------------------
@dataclass
class Case:
    id: int
    k: str
    owner: str
    priority: int
    created_hour: int
    norm_hours: float
    remaining_hours: float
    escalated: bool = False
    escalated_hour: Optional[int] = None
    closed_hour: Optional[int] = None
    meta: dict = field(default_factory=dict)


# -----------------------------
# Scenario / parameters
# -----------------------------
@dataclass
class StateParams:
    seed: int = 42
    dt_hours: int = 1
    hours_per_day: int = 24
    T_days: int = 120
    # Background volatility (small shocks strength)
    event_severity_scale: float = 0.6
    # --- sensitivity knobs ---
    throughput_mult: float = 1.0   # baseline

    # Optional scripted shocks by day (1-based). Example:
    # {30: {'SUPPLY': True, 'SANCTIONS_REGIME': 1}, 55: {'OUTBREAK': True, 'OUTBREAK_SEV': 0.8}}
    scripted_shocks: Dict[int, Dict[str, float]] = field(default_factory=dict)

    # WIP limits by role
    WIP_limits: Dict[str, int] = field(default_factory=lambda: {"Econ": 6, "Health": 6, "Finance": 6, "Head": 4})
    escalation_factor: float = 2.0

    # Governance review
    governance_mode: str = "high_latency"  # high_latency | low_latency
    PDCA_PERIOD_HOURS: int = 72  # slower review in classical bureaucracy
    NOOCRATIC_BASE_PERIOD_HOURS: int = 6
    NOOCRATIC_MIN_PERIOD_HOURS: int = 2
    NOOCRATIC_MAX_PERIOD_HOURS: int = 12

    # Trigger thresholds
    infl_hi: float = 0.06
    infl_lo: float = 0.03
    fiscal_bad: float = -0.04
    debt_bad: float = 1.0
    mu_bad: float = 0.90
    cap_bad: float = 0.50
    info_bad: float = 0.55

    # Routine workload
    routine_lambda_base: Dict[str, float] = field(default_factory=lambda: {"Econ": 1.2, "Finance": 1.1, "Health": 1.4})

    # IEKV weights (kept identical to v0 defaults)
    iekv_weights: Dict[str, float] = field(default_factory=lambda: {"O": 1.0, "W": 0.5, "E": 0.8, "A": 1.0})

    # HDI+ weights (v1)
    wY: float = 0.35
    wL: float = 0.25
    wS: float = 0.20
    wPi: float = 0.10  # inflation penalty
    wMu: float = 0.10  # pressure penalty


# -----------------------------
# IEKV computation
# -----------------------------
def compute_iekv(
    df_cases: pd.DataFrame,
    df_wip: pd.DataFrame,
    wip_limits: Dict[str, int],
    w: Optional[Dict[str, float]] = None,
    window_hours: Optional[int] = None,
) -> Dict[str, float]:
    base_w = {"O": 1.0, "W": 0.5, "E": 0.8, "A": 1.0}
    w_in = w or {}
    w_norm = {str(k).upper(): float(v) for k, v in w_in.items()}
    w = {**base_w, **w_norm}

    c = df_cases.copy()
    wip = df_wip.copy()

    if window_hours is not None and window_hours > 0 and len(wip) > 0:
        max_hour = int(wip["hour"].max())
        min_hour = max(0, max_hour - int(window_hours))
        wip = wip[wip["hour"] >= min_hour].copy()
        if len(c) > 0 and "closed_hour" in c.columns:
            c = c[(c["closed_hour"].fillna(10**18).astype(int) >= min_hour)].copy()

    # W: excess WIP over limits
    W = 0.0
    if len(wip) > 0:
        for role, lim in wip_limits.items():
            if role in wip.columns:
                W += float(np.maximum(0.0, wip[role].to_numpy() - lim).mean())

    # O: overload tail proxy (p90 of total WIP)
    O = 0.0
    if len(wip) > 0:
        cols = [r for r in wip_limits.keys() if r in wip.columns]
        if cols:
            tot = wip[cols].sum(axis=1).to_numpy()
            O = float(np.percentile(tot, 90))

    # E: escalation rate
    E = 0.0
    if len(c) > 0 and "escalated" in c.columns:
        E = float(np.mean(c["escalated"].fillna(False).astype(bool)))

    # A: agentity proxy (1 - escalation rate) in v1
    A = 1.0 - E

    total_lim = float(sum(wip_limits.values())) if wip_limits else 1.0
    if total_lim <= 0:
        total_lim = 1.0
    O_n = O / total_lim
    W_n = W / total_lim

    IEKV = w["O"] * O_n + w["W"] * W_n + w["E"] * E + w["A"] * (1.0 - A)

    return {"IEKV": float(IEKV), "O": float(O_n), "W": float(W_n), "E": float(E), "A": float(A)}


# -----------------------------
# v1 Sandbox
# -----------------------------
class StateSandboxV1:
    """
    v1 state = compact closed subgraph (15 vars):
      Economy: Y, K, I, Inflation, Profitability
      Governance: Gov_capacity, Policy_coherence, Legitimacy, Transaction_costs, Fiscal_balance, Public_debt
      External: Supply_chain_integrity, Sanctions_regime
      Info: Info_integrity
      Complexity loop: Omega, Omega_eff, kappa_star, L_star, mu
    """
    def __init__(self, params: StateParams):
        self.p = params
        self.rng = np.random.default_rng(self.p.seed)
        self.hour = 0
        self.case_seq = 0

        self.scripted_shocks = dict(self.p.scripted_shocks) if self.p.scripted_shocks else {}

        # lightweight rolling buffers for fast IEKV computation (avoid pandas in-loop)
        self._wip_tot_hist: List[float] = []
        self._wip_excess_hist: List[float] = []

        # cases and queues
        self.cases: Dict[int, Case] = {}
        self.role_queues: Dict[str, List[int]] = {r: [] for r in ["Econ", "Health", "Finance", "Head"]}
        self.deferred_queues: Dict[str, List[Tuple[int, int]]] = {r: [] for r in ["Econ", "Health", "Finance"]}

        # governance review scheduling
        self._next_review_hour = 0
        self._noocratic_period = float(self.p.NOOCRATIC_BASE_PERIOD_HOURS)

        # --- macro state (v1) ---
        # Economy
        self.Y = 0.75
        self.K = 1.00
        self.I = 0.20
        self.Inflation = 0.04
        self.Profitability = 1.00

        # Governance
        self.Gov_capacity = 0.65
        self.Policy_coherence = 0.60
        self.Legitimacy = 0.60
        self.Transaction_costs = 1.00
        self.Fiscal_balance = -0.02
        self.Public_debt = 0.80

        # External
        self.Supply_chain_integrity = 0.80
        self.Sanctions_regime = 0  # 0/1/2

        # Info
        self.Info_integrity = 0.60

        # Complexity loop
        self.Omega = 1.00
        self.Omega_eff = 0.85
        self.kappa_star = 0.70
        self.L_star = 0.70
        self.mu = 0.25

        # knobs (flat, ScenarioKnobs-like)
        self.knobs: Dict[str, float] = {
            "external.supply_chain_delta": 0.0,
            "external.sanctions_delta": 0.0,

            "monetary.rate_stance": 0.0,
            "monetary.fx_stabilization_effort": 0.0,

            "fiscal.spend_stimulus": 0.0,
            "fiscal.austerity": 0.0,
            "fiscal.tax_enforcement": 0.0,
            "fiscal.debt_management": 0.0,

            "governance.coherence_boost": 0.0,
            "governance.tc_reduction_program": 0.0,
            "governance.capacity_surge": 0.0,

            "info.info_integrity_boost": 0.0,

            "health.npi_intensity": 0.0,
            "health.health_mobilization": 0.0,
        }
        self.target_mod: Dict[str, float] = {"kappa_star_shift": 0.0, "L_star_shift": 0.0}

        # decay registry (key -> half-life days)
        self._half_life: Dict[str, float] = {}

        # logs
        self.log_cases: List[dict] = []
        self.log_wip: List[dict] = []
        self.log_state: List[dict] = []
        self.log_events: List[dict] = []

    # ------------------------
    # Case mechanics
    # ------------------------
    def _new_case_id(self) -> int:
        self.case_seq += 1
        return self.case_seq

    def _enqueue_case(self, k: str, owner: str, norm_hours: float, priority: int, meta: Optional[dict] = None):
        cid = self._new_case_id()
        c = Case(
            id=cid, k=k, owner=owner, priority=priority,
            created_hour=self.hour, norm_hours=float(norm_hours),
            remaining_hours=float(norm_hours), meta=meta or {}
        )

        # WIP deferral for non-Head roles
        if owner != "Head":
            lim = int(self.p.WIP_limits.get(owner, 6))
            wip_now = len(self.role_queues[owner]) + len(self.deferred_queues[owner])
            if wip_now >= lim:
                # defer (cid, release_hour)
                release = self.hour + self.p.hours_per_day
                self.deferred_queues[owner].append((cid, release))
            else:
                self.role_queues[owner].append(cid)
        else:
            self.role_queues["Head"].append(cid)

        self.cases[cid] = c

    def _release_deferred(self):
        for role, dq in self.deferred_queues.items():
            if not dq:
                continue
            ready = [(cid, rh) for (cid, rh) in dq if rh <= self.hour]
            self.deferred_queues[role] = [(cid, rh) for (cid, rh) in dq if rh > self.hour]
            for cid, _ in ready:
                self.role_queues[role].append(cid)

    def _escalate_cases(self):
        # escalate when waiting too long
        for role in ["Econ", "Health", "Finance"]:
            if not self.role_queues[role]:
                continue
            cid = self.role_queues[role][0]
            c = self.cases[cid]
            if c.escalated:
                continue
            wait_hours = self.hour - c.created_hour
            if wait_hours > self.p.escalation_factor * c.norm_hours:
                c.escalated = True
                c.escalated_hour = self.hour
                # move to Head
                self.role_queues[role].pop(0)
                self.role_queues["Head"].append(cid)

    def _work_on_role(self, role: str):
        if not self.role_queues[role]:
            return
        cid = self.role_queues[role][0]
        c = self.cases[cid]
        base_quanta = 1.0
        if self.p.governance_mode == "low_latency":
            base_quanta = 2.0  # automation/AI assistance: more throughput per hour
            
        quanta = base_quanta * self.p.throughput_mult  # sensitivity modifier
        
        c.remaining_hours -= quanta
        if c.remaining_hours <= 0:
            c.closed_hour = self.hour
            self.role_queues[role].pop(0)
            self._actuate_on_close(c)
            # log case closure
            self.log_cases.append({
                "id": c.id,
                "k": c.k,
                "owner": c.owner,
                "created_hour": c.created_hour,
                "closed_hour": c.closed_hour,
                "norm_hours": c.norm_hours,
                "escalated": c.escalated,
            })

    def _process_roles(self):
        self._release_deferred()
        # simple round-robin per hour
        for role in ["Econ", "Health", "Finance", "Head"]:
            self._work_on_role(role)

    def _actuate_on_close(self, c: Case):
        """
        Apply knob deltas/target shifts on case close.
        meta schema:
          - knob_deltas: {key: delta}
          - target_deltas: {key: delta}
          - half_life_days: {key: hl}
        """
        kd: Dict[str, float] = dict(c.meta.get("knob_deltas", {}) or {})
        td: Dict[str, float] = dict(c.meta.get("target_deltas", {}) or {})
        hl: Dict[str, float] = dict(c.meta.get("half_life_days", {}) or {})

        for k, d in kd.items():
            if k in self.knobs:
                self.knobs[k] = float(self.knobs[k] + float(d))
            else:
                self.knobs[k] = float(d)
            if k in hl:
                self._half_life[k] = float(hl[k])

        for k, d in td.items():
            if k in self.target_mod:
                self.target_mod[k] = float(self.target_mod[k] + float(d))
            else:
                self.target_mod[k] = float(d)
            if k in hl:
                self._half_life[k] = float(hl[k])

    # ------------------------
    # Events
    # ------------------------
    def _daily_event_stream(self) -> Dict[str, Any]:
        """
        Returns dict with booleans and severities.
        Event intensities depend weakly on state (mu, sanctions, supply chain).
        """
        # base probabilities
        p_outbreak = 0.02 + 0.03 * clamp(self.mu / 1.2, 0, 1)
        p_supply = 0.02 + 0.04 * (1.0 - self.Supply_chain_integrity) + 0.02 * (self.Sanctions_regime / 2)
        p_revenue = 0.02 + 0.03 * clamp((self.Public_debt - 0.8) / 0.6, 0, 1)
        p_info = 0.015 + 0.04 * clamp((0.65 - self.Info_integrity) / 0.4, 0, 1)

        OUTBREAK = bool(self.rng.random() < p_outbreak)
        SUPPLY = bool(self.rng.random() < p_supply)
        REVENUE = bool(self.rng.random() < p_revenue)
        INFO = bool(self.rng.random() < p_info)

        # sev = float(self.rng.uniform(0.4, 1.0)) if OUTBREAK else 0.0
        sev = float(self.rng.uniform(0.4, 1.0)) * self.p.event_severity_scale if OUTBREAK else 0.0

        # Scripted shocks override (day is 1-based)
        day = self.hour // self.p.hours_per_day + 1
        if day in self.scripted_shocks:
            sc = self.scripted_shocks[day]
            # Override event booleans if provided
            if "OUTBREAK" in sc: OUTBREAK = bool(sc["OUTBREAK"])
            if "SUPPLY" in sc: SUPPLY = bool(sc["SUPPLY"])
            if "REVENUE" in sc: REVENUE = bool(sc["REVENUE"])
            if "INFO" in sc: INFO = bool(sc["INFO"])
            # Override severity
            if "OUTBREAK_SEV" in sc and OUTBREAK:
                sev = float(sc["OUTBREAK_SEV"])
            # Persistent external regime toggles
            if "SANCTIONS_REGIME" in sc:
                self.Sanctions_regime = int(sc["SANCTIONS_REGIME"])
            if "SUPPLY_CHAIN_DELTA" in sc:
                self.knobs["external.supply_chain_delta"] += float(sc["SUPPLY_CHAIN_DELTA"])
                # default persistence if not set by other shocks
                self._half_life["external.supply_chain_delta"] = float(sc.get("SUPPLY_CHAIN_HALF_LIFE", 60.0))
        return {"OUTBREAK": OUTBREAK, "SUPPLY": SUPPLY, "REVENUE": REVENUE, "INFO": INFO, "severity": sev}

    def _apply_exogenous_shocks_to_knobs(self, ev: Dict[str, Any]):
        # supply shocks reduce supply chain integrity for a while
        if ev.get("SUPPLY", False):
            # self.knobs["external.supply_chain_delta"] -= 0.08
            self.knobs["external.supply_chain_delta"] -= 0.08 * self.p.event_severity_scale

            self._half_life["external.supply_chain_delta"] = 30.0
        # revenue shock increases austerity pressure demand (not direct policy; adds background pressure)
        if ev.get("REVENUE", False):
            # treat as temporary fiscal stress that pushes mu via debt/fiscal dynamics (handled in state update)
            pass
        # info attack reduces info integrity unless countered
        if ev.get("INFO", False):
            self.Info_integrity = clamp(self.Info_integrity - 0.05, 0.0, 1.0)

        # outbreak creates health-related policy demand, handled via cases

    # ------------------------
    # Triggers -> cases
    # ------------------------
    def _issue_cases_from_triggers(self, ev: Dict[str, Any]):
        # Monetary
        # if self.Inflation > self.p.infl_hi:
        # гистерезис: инфляция сама по себе — не кризис, кризис = инфляция + управленческое давление
        if self.Inflation > self.p.infl_hi and self.mu > 0.6:    
            self._enqueue_case(
                "MONETARY_TIGHTEN", "Econ", norm_hours=24, priority=3,
                meta={
                    "knob_deltas": {"monetary.rate_stance": 0.25, "monetary.fx_stabilization_effort": 0.10},
                    "half_life_days": {"monetary.rate_stance": 45.0, "monetary.fx_stabilization_effort": 30.0}
                }
            )
        elif self.Inflation < self.p.infl_lo and self.Y < 0.75:
            self._enqueue_case(
                "MONETARY_EASE", "Econ", norm_hours=24, priority=3,
                meta={"knob_deltas": {"monetary.rate_stance": -0.25}, "half_life_days": {"monetary.rate_stance": 45.0}}
            )

        # Fiscal
        if self.Fiscal_balance < self.p.fiscal_bad:
            self._enqueue_case(
                "FISCAL_AUSTERITY", "Finance", norm_hours=36, priority=3,
                meta={
                    "knob_deltas": {"fiscal.austerity": 0.20},
                    "target_deltas": {"L_star_shift": -0.05},
                    "half_life_days": {"fiscal.austerity": 60.0, "L_star_shift": 90.0}
                }
            )
            self._enqueue_case(
                "TAX_ENFORCEMENT", "Finance", norm_hours=48, priority=2,
                meta={
                    "knob_deltas": {"fiscal.tax_enforcement": 0.15, "governance.tc_reduction_program": 0.05},
                    "half_life_days": {"fiscal.tax_enforcement": 120.0, "governance.tc_reduction_program": 180.0}
                }
            )

        if self.Public_debt > self.p.debt_bad:
            self._enqueue_case(
                "DEBT_MANAGEMENT", "Finance", norm_hours=48, priority=2,
                meta={"knob_deltas": {"fiscal.debt_management": 0.20}, "half_life_days": {"fiscal.debt_management": 90.0}}
            )

        # Info integrity
        if self.Info_integrity < self.p.info_bad:
            self._enqueue_case(
                "INFO_INTEGRITY_CAMPAIGN", "Head", norm_hours=60, priority=2,
                meta={"knob_deltas": {"info.info_integrity_boost": 0.15}, "half_life_days": {"info.info_integrity_boost": 120.0}}
            )

        # Governance overload
        # if self.mu > self.p.mu_bad or self.Gov_capacity < self.p.cap_bad:
        # гистерезис:
        if (self.mu > self.p.mu_bad and self.Gov_capacity < self.kappa_star):
            self._enqueue_case(
                "CAPACITY_SURGE", "Head", norm_hours=12, priority=4,
                meta={
                    "knob_deltas": {"governance.capacity_surge": 0.25},
                    "target_deltas": {"kappa_star_shift": 0.10},
                    "half_life_days": {"governance.capacity_surge": 30.0, "kappa_star_shift": 60.0}
                }
            )
            self._enqueue_case(
                "GOV_COHERENCE_REFORM", "Head", norm_hours=72, priority=3,
                meta={
                    "knob_deltas": {"governance.coherence_boost": 0.15, "governance.tc_reduction_program": 0.10},
                    "half_life_days": {"governance.coherence_boost": 180.0, "governance.tc_reduction_program": 180.0}
                }
            )

        # Health events as cases
        if ev.get("OUTBREAK", False):
            self._enqueue_case(
                "HEALTH_NPI", "Health", norm_hours=24, priority=3,
                meta={
                    "knob_deltas": {"health.npi_intensity": 0.25},
                    "target_deltas": {"L_star_shift": -0.03},
                    "half_life_days": {"health.npi_intensity": 21.0, "L_star_shift": 45.0}
                }
            )
            self._enqueue_case(
                "HEALTH_MOBILIZATION", "Health", norm_hours=36, priority=3,
                meta={
                    "knob_deltas": {"health.health_mobilization": 0.25, "fiscal.spend_stimulus": 0.10},
                    "half_life_days": {"health.health_mobilization": 30.0, "fiscal.spend_stimulus": 45.0}
                }
            )
            self._enqueue_case(
                "FISCAL_STIMULUS", "Finance", norm_hours=36, priority=3,
                meta={
                    "knob_deltas": {"fiscal.spend_stimulus": 0.20},
                    "target_deltas": {"L_star_shift": 0.05},
                    "half_life_days": {"fiscal.spend_stimulus": 60.0, "L_star_shift": 90.0}
                }
            )

    # ------------------------
    # Knobs decay and application
    # ------------------------
    def _decay_knobs_daily(self):
        # apply exponential decay to every knob/target that has half-life
        for key, hl in list(self._half_life.items()):
            f = exp_decay_factor(hl, 1.0)
            if key in self.knobs:
                self.knobs[key] *= f
            elif key in self.target_mod:
                self.target_mod[key] *= f
            # if almost gone, drop
            if (key in self.knobs and abs(self.knobs[key]) < 1e-4) or (key in self.target_mod and abs(self.target_mod[key]) < 1e-4):
                self._half_life.pop(key, None)

        # apply info campaign to info_integrity gradually
        boost = self.knobs.get("info.info_integrity_boost", 0.0)
        if boost != 0.0:
            self.Info_integrity = clamp(self.Info_integrity + 0.01 * boost, 0.0, 1.0)

    # ------------------------
    # Macro update (compact closed subgraph)
    # ------------------------
    def _update_state_daily(self, iekv_day: float, ev: Dict[str, Any]):
        # unpack knobs
        rate = self.knobs["monetary.rate_stance"]
        fx_eff = self.knobs["monetary.fx_stabilization_effort"]

        stim = self.knobs["fiscal.spend_stimulus"]
        aust = self.knobs["fiscal.austerity"]
        tax = self.knobs["fiscal.tax_enforcement"]
        debt_mgmt = self.knobs["fiscal.debt_management"]

        coh_boost = self.knobs["governance.coherence_boost"]
        tc_prog = self.knobs["governance.tc_reduction_program"]
        cap_surge = self.knobs["governance.capacity_surge"]

        npi = self.knobs["health.npi_intensity"]
        hm = self.knobs["health.health_mobilization"]

        # external knobs affect supply chain and sanctions regime
        self.Supply_chain_integrity = clamp(
            self.Supply_chain_integrity
            + 0.01 * (0.80 - self.Supply_chain_integrity)
            + float(self.knobs.get("external.supply_chain_delta", 0.0))
            + 0.01 * fx_eff,
            0.0, 1.0
        )

        # sanctions as integer-ish with drift (kept simple)
        sanc_delta = float(self.knobs.get("external.sanctions_delta", 0.0))
        if sanc_delta != 0:
            self.Sanctions_regime = int(clamp(self.Sanctions_regime + int(round(sanc_delta)), 0, 2))

        # --- Complexity loop ---
        # Omega grows with sanctions, supply fragility, debt (proxy)
        self.Omega = clamp(
            1.0 + 0.35 * (self.Sanctions_regime / 2.0) + 0.60 * (1.0 - self.Supply_chain_integrity) + 0.25 * clamp(self.Public_debt - 0.8, 0, 1),
            0.0, 2.0
        )
        # Info integrity reduces effective complexity
        self.Omega_eff = clamp(self.Omega * (1.0 - 0.30 * clamp(self.Info_integrity - 0.5, -0.5, 0.5)), 0.0, 2.0)

        self.kappa_star = clamp(0.60 + 0.45 * self.Omega_eff + float(self.target_mod.get("kappa_star_shift", 0.0)), 0.0, 1.5)
        self.L_star = clamp(0.60 + 0.30 * self.Omega_eff + float(self.target_mod.get("L_star_shift", 0.0)), 0.0, 1.5)

        gapK = max(0.0, self.kappa_star - self.Gov_capacity)
        gapL = max(0.0, self.L_star - self.Legitimacy)
        mu_target = gapK + 0.8 * gapL
        # self.mu = clamp(0.20 * self.mu + 0.80 * mu_target, 0.0, 2.0)
        # делаем давление менее “нервным”
        self.mu = clamp(0.40 * self.mu + 0.60 * mu_target, 0.0, 2.0)


        # --- Governance block ---
        # Transaction costs rise with pressure, fall with programs/coherence
        self.Transaction_costs = clamp(
            1.0 + 0.60 * self.mu - 0.35 * tc_prog - 0.15 * coh_boost,
            0.6, 2.0
        )

        # policy coherence improves with reforms, degrades under pressure
        self.Policy_coherence = clamp(
            0.60 + 0.30 * coh_boost + 0.15 * clamp(self.Info_integrity - 0.5, -0.5, 0.5) - 0.25 * clamp(self.mu / 1.5, 0, 1),
            0.0, 1.0
        )

        # governance capacity: recovers toward 0.70, boosted by surge, fatigued by mu and IEKV
        self.Gov_capacity = clamp(
            self.Gov_capacity
            + 0.01 * (0.70 - self.Gov_capacity)
            + 0.02 * cap_surge
            - 0.015 * clamp(self.mu / 1.5, 0, 1)
            - 0.010 * clamp(iekv_day / 10.0, 0, 2),
            0.0, 1.0
        )

        # --- Economy block ---
        # Profitability falls with transaction costs and inflation, rises with Y
        self.Profitability = clamp(1.0 + 0.25 * (self.Y - 0.75) - 0.45 * (self.Transaction_costs - 1.0) - 0.20 * self.Inflation, 0.0, 1.5)

        # Investment reacts to profitability, monetary stance, stimulus
        self.I = clamp(
            self.I + 0.05 * (self.Profitability - self.I) - 0.03 * rate + 0.02 * stim,
            0.0, 0.5
        )

        # Capital accumulation with depreciation and supply frictions
        self.K = clamp(
            self.K + 0.02 * (self.I - 0.03) - 0.01 * (1.0 - self.Supply_chain_integrity),
            0.5, 2.0
        )

        # Output mean-reverts to a production proxy, penalized by mu and NPI
        prod_proxy = (0.65 * self.K + 0.35 * (self.I / 0.20)) / (1.0 + 0.50 * max(0.0, self.Transaction_costs - 1.0))
        prod_proxy = clamp(prod_proxy, 0.0, 1.2)
        self.Y = clamp(
            self.Y + 0.08 * (prod_proxy - self.Y) - 0.05 * npi - 0.03 * clamp(self.mu / 1.5, 0, 1),
            0.0, 1.2
        )

        # Inflation reacts to supply frictions, sanctions, stimulus; reduced by rate stance
        self.Inflation = clamp(
            self.Inflation
            + 0.03 * (1.0 - self.Supply_chain_integrity)
            + 0.02 * (self.Sanctions_regime / 2.0)
            + 0.015 * stim
            - 0.04 * rate
            + 0.01 * (self.Inflation - 0.04) * 0.0,  # placeholder for momentum
            0.0, 0.30
        )

        # --- Fiscal block ---
        # Fiscal balance: improves with tax enforcement/austerity, worsens with stimulus and health mobilization
        self.Fiscal_balance = clamp(
            self.Fiscal_balance
            + 0.05 * (tax - 0.0)
            + 0.05 * aust
            - 0.06 * stim
            - 0.04 * hm
            + 0.02 * (self.Y - 0.75),
            -0.25, 0.15
        )

        # Public debt accumulates with deficits; debt_management reduces drift
        self.Public_debt = clamp(
            self.Public_debt + max(0.0, -self.Fiscal_balance) * (1.0 - 0.40 * debt_mgmt),
            0.0, 2.0
        )

        # Legitimacy: rises with Y and coherence, falls with inflation, austerity and NPI; pressure also erodes
        self.Legitimacy = clamp(
            self.Legitimacy
            + 0.02 * (self.Y - 0.75)
            + 0.02 * (self.Policy_coherence - 0.60)
            - 0.03 * self.Inflation
            - 0.02 * aust
            - 0.015 * npi
            - 0.01 * clamp(self.mu / 1.5, 0, 1)
            + 0.01 * stim,
            0.0, 1.0
        )

    # ------------------------
    # HDI+
    # ------------------------
    def hdi_plus(self) -> float:
        social_proxy = 0.5 * self.Policy_coherence + 0.5 * self.Info_integrity
        return (
            self.p.wY * clamp(self.Y, 0.0, 1.0)
            + self.p.wL * self.Legitimacy
            + self.p.wS * clamp(social_proxy, 0.0, 1.0)
            - self.p.wPi * self.Inflation
            - self.p.wMu * clamp(self.mu / 2.0, 0.0, 1.0)
        )

    def loop_modes(self) -> Dict[str, str]:
        # E if there are open cases in domain or deferrals; else A
        def has_open(prefix: str) -> bool:
            for c in self.cases.values():
                if c.closed_hour is None and c.k == prefix:
                    return True
            return False
        # map: treat monetary cases as INF, health as HEA, fiscal as BUD
        INF = "E" if (has_open("MONETARY_TIGHTEN") or has_open("MONETARY_EASE") or len(self.deferred_queues["Econ"]) > 0) else "A"
        HEA = "E" if (has_open("HEALTH_NPI") or has_open("HEALTH_MOBILIZATION") or len(self.deferred_queues["Health"]) > 0) else "A"
        BUD = "E" if (has_open("FISCAL_AUSTERITY") or has_open("FISCAL_STIMULUS") or has_open("DEBT_MANAGEMENT") or len(self.deferred_queues["Finance"]) > 0) else "A"
        return {"INF": INF, "HEA": HEA, "BUD": BUD}


    def _compute_iekv_fast(self, window_hours: int) -> Dict[str, float]:
        """Fast IEKV proxy from rolling per-hour WIP history and recent case closures.
        Avoids pandas inside the simulation loop.
        """
        base_w = {"O": 1.0, "W": 0.5, "E": 0.8, "A": 1.0}
        w_in = self.p.iekv_weights or {}
        w_norm = {str(k).upper(): float(v) for k, v in w_in.items()}
        w = {**base_w, **w_norm}

        # window slice over per-hour history
        n = int(max(1, window_hours))
        tot_hist = np.array(self._wip_tot_hist[-n:], dtype=float) if self._wip_tot_hist else np.array([0.0], dtype=float)
        exc_hist = np.array(self._wip_excess_hist[-n:], dtype=float) if self._wip_excess_hist else np.array([0.0], dtype=float)

        O = float(np.percentile(tot_hist, 90)) if len(tot_hist) > 0 else 0.0
        W = float(np.mean(exc_hist)) if len(exc_hist) > 0 else 0.0

        # escalation rate over cases closed in the window
        min_hour = max(0, self.hour - n)
        closed = [c for c in self.cases.values() if c.closed_hour is not None and int(c.closed_hour) >= min_hour]
        if closed:
            E = float(np.mean([bool(c.escalated) for c in closed]))
        else:
            E = 0.0
        A = 1.0 - E

        total_lim = float(sum(self.p.WIP_limits.values())) if self.p.WIP_limits else 1.0
        if total_lim <= 0:
            total_lim = 1.0
        O_n = O / total_lim
        W_n = W / total_lim

        IEKV = w["O"] * O_n + w["W"] * W_n + w["E"] * E + w["A"] * (1.0 - A)
        return {"IEKV": float(IEKV), "O": float(O_n), "W": float(W_n), "E": float(E), "A": float(A)}

    # ------------------------
    # Governance review (decides nothing directly; just scheduling here)
    # ------------------------
    def _review_policy(self):
        # In v1, "review" primarily changes responsiveness (noocratic adjusts review period based on overload)
        # Using WIP as a simple proxy
        wip_total = (len(self.role_queues["Econ"]) + len(self.deferred_queues["Econ"])
                     + len(self.role_queues["Health"]) + len(self.deferred_queues["Health"])
                     + len(self.role_queues["Finance"]) + len(self.deferred_queues["Finance"])
                     + len(self.role_queues["Head"]))
        # noocratic: shorten period when overloaded
        if self.p.governance_mode == "low_latency":
            target = clamp(self.p.NOOCRATIC_BASE_PERIOD_HOURS - 0.5 * wip_total, self.p.NOOCRATIC_MIN_PERIOD_HOURS, self.p.NOOCRATIC_MAX_PERIOD_HOURS)
            self._noocratic_period = 0.7 * self._noocratic_period + 0.3 * target

    # ------------------------
    # main loop
    # ------------------------
    def step(self):
        # review scheduling
        if self.p.governance_mode == "high_latency":
            if self.hour >= self._next_review_hour:
                self._review_policy()
                self._next_review_hour += self.p.PDCA_PERIOD_HOURS
        else:
            if self.hour >= self._next_review_hour:
                self._review_policy()
                self._next_review_hour += max(1, int(self._noocratic_period))

        # hourly processing
        self._escalate_cases()
        self._process_roles()

        
        # update rolling WIP hist (per-hour) for fast IEKV
        wip_econ = len(self.role_queues.get("Econ", [])) + len(self.deferred_queues.get("Econ", []))
        wip_health = len(self.role_queues.get("Health", [])) + len(self.deferred_queues.get("Health", []))
        wip_fin = len(self.role_queues.get("Finance", [])) + len(self.deferred_queues.get("Finance", []))
        wip_head = len(self.role_queues.get("Head", [])) + len(self.deferred_queues.get("Head", []))
        tot_wip = float(wip_econ + wip_health + wip_fin + wip_head)
        excess = 0.0
        excess += max(0.0, wip_econ - self.p.WIP_limits.get("Econ", 0))
        excess += max(0.0, wip_health - self.p.WIP_limits.get("Health", 0))
        excess += max(0.0, wip_fin - self.p.WIP_limits.get("Finance", 0))
        excess += max(0.0, wip_head - self.p.WIP_limits.get("Head", 0))
        self._wip_tot_hist.append(tot_wip)
        self._wip_excess_hist.append(float(excess))
# hourly WIP log
        self.log_wip.append({
            "hour": self.hour,
            "Econ": len(self.role_queues["Econ"]) + len(self.deferred_queues["Econ"]),
            "Health": len(self.role_queues["Health"]) + len(self.deferred_queues["Health"]),
            "Finance": len(self.role_queues["Finance"]) + len(self.deferred_queues["Finance"]),
            "Head": len(self.role_queues["Head"]),
        })

        # daily tick at start of day (hour%24==0 and hour>0)
        if (self.hour % self.p.hours_per_day) == 0 and self.hour > 0:
            day = now_day(self.hour, self.p.hours_per_day)

            # events
            ev = self._daily_event_stream()
            self.log_events.append({"day": day, **ev})
            self._apply_exogenous_shocks_to_knobs(ev)

            # routine governance workload scales with pressure mu
            for role, lam0 in self.p.routine_lambda_base.items():
                # lam = float(lam0 * (1.0 + 0.8 * clamp(self.mu / 1.2, 0, 1)))
                # смягчаем фоновую рутину
                lam = float(lam0 * (0.5 + 0.6 * clamp(self.mu / 1.2, 0, 1)))
                n = int(self.rng.poisson(lam))
                for _ in range(n):
                    # generic routine case: coherence reform microtask (cheap, non-gated)
                    self._enqueue_case(
                        "ROUTINE", role, norm_hours=4.0, priority=1,
                        meta={"knob_deltas": {"governance.coherence_boost": 0.01}, "half_life_days": {"governance.coherence_boost": 60.0}}
                    )

            # trigger-driven cases
            self._issue_cases_from_triggers(ev)

            # compute 7d IEKV
            iekv_parts = self._compute_iekv_fast(window_hours=7 * self.p.hours_per_day)
            iekv_day = iekv_parts["IEKV"]

            # knobs decay and gradual effects
            self._decay_knobs_daily()

            # update state via v1 macro subgraph
            self._update_state_daily(iekv_day, ev)

            # log state
            modes = self.loop_modes()
            self.log_state.append({
                "day": day,
                # state vars
                "Y": self.Y, "K": self.K, "I": self.I, "Inflation": self.Inflation, "Profitability": self.Profitability,
                "Gov_capacity": self.Gov_capacity, "Policy_coherence": self.Policy_coherence, "Legitimacy": self.Legitimacy,
                "Transaction_costs": self.Transaction_costs, "Fiscal_balance": self.Fiscal_balance, "Public_debt": self.Public_debt,
                "Supply_chain_integrity": self.Supply_chain_integrity, "Sanctions_regime": self.Sanctions_regime, "Info_integrity": self.Info_integrity,
                "Omega": self.Omega, "Omega_eff": self.Omega_eff, "kappa_star": self.kappa_star, "L_star": self.L_star, "mu": self.mu,
                # metrics
                "HDI_plus": self.hdi_plus(),
                "IEKV_7d": iekv_day, "O": iekv_parts["O"], "W": iekv_parts["W"], "E": iekv_parts["E"], "A": iekv_parts["A"],
                # modes
                **modes,
            })

        self.hour += self.p.dt_hours

    def run(self) -> Dict[str, pd.DataFrame]:
        total_hours = int(self.p.T_days * self.p.hours_per_day)
        for _ in range(total_hours):
            self.step()
        return {
            "state": pd.DataFrame(self.log_state),
            "wip": pd.DataFrame(self.log_wip),
            "cases": pd.DataFrame(self.log_cases),
            "events": pd.DataFrame(self.log_events),
        }


# -----------------------------
# experiment runner
# -----------------------------
def run_experiment(
    modes: Tuple[str, ...] = ("high_latency", "low_latency"),
    T_days: int = 120,
    seed: int = 42,
    scripted_shocks: Dict[int, Dict[str, float]] | None = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    for m in modes:
        p = StateParams(seed=seed, governance_mode=m, T_days=T_days, scripted_shocks=(scripted_shocks or {}))
        sim = StateSandboxV1(p)
        out[m] = sim.run()
    return out

# Prepare data
def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Ensure day column exists
    if "day" not in df.columns:
        # infer: state log is per day in this model; create index-based day
        df["day"] = np.arange(1, len(df)+1)
    # E_share per day: fraction of loops in escalation
    loops = ["INF","HEA","BUD"]
    df["E_share"] = (df[loops] == "E").mean(axis=1)
    return df

def run_monte_carlo(
    seeds: List[int],
    modes: Tuple[str, ...] = ("high_latency", "low_latency"),
    T_days: int = 120,
    scripted_shocks: Dict[int, Dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Run a Monte Carlo sweep across seeds. Returns summary rows per (mode, seed)."""
    rows: List[Dict[str, float]] = []
    for s in seeds:
        out = run_experiment(modes=modes, T_days=T_days, seed=s, scripted_shocks=scripted_shocks)
        for m in modes:
            st = out[m]["state"].copy()
            # E_share: fraction of loops in escalation (E)
            st["E_share"] = (st[["INF", "HEA", "BUD"]] == "E").mean(axis=1)
            rows.append({
                "seed": s,
                "mode": m,
                "avg_HDI_plus": float(st["HDI_plus"].mean()),
                "avg_IEKV_7d": float(st["IEKV_7d"].mean()),
                "avg_E_share": float(st["E_share"].mean()),
                "p90_IEKV_7d": float(st["IEKV_7d"].quantile(0.90)),
                "min_HDI_plus": float(st["HDI_plus"].min()),
            })
    return pd.DataFrame(rows)

from dataclasses import replace
from typing import Callable

def run_sensitivity(
    base_params: StateParams,
    param_name: str,
    values: List[float],
    seeds: List[int],
    modes: Tuple[str, ...] = ("high_latency", "low_latency"),
    T_days: int = 120,
    scripted_shocks: Dict[int, Dict[str, float]] | None = None,
) -> pd.DataFrame:
    """
    One-factor sensitivity: sweep param_name over values, run MC over seeds, for each governance mode.

    Returns tidy DF:
      mode, seed, param_name, param_value,
      avg_HDI_plus, avg_IEKV_7d, avg_E_share, p90_IEKV_7d, min_HDI_plus
    """
    if not hasattr(base_params, param_name):
        raise AttributeError(f"StateParams has no attribute '{param_name}'")

    rows: List[Dict[str, float]] = []

    for v in values:
        for s in seeds:
            for m in modes:
                # clone params safely
                p = replace(base_params)
                p.seed = s
                p.governance_mode = m
                p.T_days = T_days
                p.scripted_shocks = (scripted_shocks or {})

                # set swept parameter
                setattr(p, param_name, float(v))

                sim = StateSandboxV1(p)
                out = sim.run()
                st = out["state"].copy()
                st["E_share"] = (st[["INF", "HEA", "BUD"]] == "E").mean(axis=1)

                rows.append({
                    "mode": m,
                    "seed": s,
                    "param_name": param_name,
                    "param_value": float(v),

                    "avg_HDI_plus": float(st["HDI_plus"].mean()),
                    "avg_IEKV_7d": float(st["IEKV_7d"].mean()),
                    "avg_E_share": float(st["E_share"].mean()),
                    "p90_IEKV_7d": float(st["IEKV_7d"].quantile(0.90)),
                    "min_HDI_plus": float(st["HDI_plus"].min()),
                })

    return pd.DataFrame(rows)

def plot_sensitivity_boxplots(
    df: pd.DataFrame,
    metric: str,
    param_name: str | None = None,
    modes_order: Tuple[str, ...] = ("high_latency", "low_latency"),
    mode_label_map: Dict[str, str] | None = None,
    value_formatter: Callable[[float], str] | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    save_path: str | None = None,
):
    """
    Universal boxplot builder for sensitivity DF.

    df must contain:
      mode, param_value (and optionally param_name), and the metric column.

    metric: e.g. "avg_IEKV_7d" or "min_HDI_plus"
    param_name: used only for labeling if you want; if None, tries to read df['param_name'].
    """
    import matplotlib.pyplot as plt

    if metric not in df.columns:
        raise KeyError(f"Metric '{metric}' not found in df columns: {list(df.columns)}")

    if "mode" not in df.columns or "param_value" not in df.columns:
        raise KeyError("df must contain 'mode' and 'param_value' columns")

    if param_name is None:
        param_name = str(df["param_name"].iloc[0]) if "param_name" in df.columns else "param"

    mode_label_map = mode_label_map or {
        "high_latency": "High-latency",
        "low_latency": "Low-latency",
    }

    if value_formatter is None:
        # default formatting: keep short
        def value_formatter(x: float) -> str:
            # integers look like 2, floats like 0.75
            if abs(x - round(x)) < 1e-9:
                return str(int(round(x)))
            return f"{x:.2g}"

    values_sorted = sorted(df["param_value"].unique())

    labels = []
    data = []

    for mode in modes_order:
        dmode = df[df["mode"] == mode]
        if dmode.empty:
            continue
        for v in values_sorted:
            subset = dmode[dmode["param_value"] == v]
            if subset.empty:
                continue
            data.append(subset[metric].values)
            labels.append(f"{mode_label_map.get(mode, mode)}\n{param_name}={value_formatter(float(v))}")

    plt.figure(figsize=(9, 4.8))
    plt.boxplot(data, labels=labels)

    if ylabel is None:
        ylabel = metric.replace("_", " ")
    plt.ylabel(ylabel)

    if title is None:
        title = f"Sensitivity of {metric} to {param_name}"
    plt.title(title)

    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    res = run_experiment()
    # for mode, logs in res.items():
    #     df = logs["state"]
    #     print(mode, "avg_HDI+", float(df["HDI_plus"].mean()), "avg_IEKV", float(df["IEKV_7d"].mean()), "E_share", float(df[["INF","HEA","BUD"]].eq("E").mean(axis=1).mean()))
    
    # dfs = {k: prep(v["state"]) for k,v in res.items()}

    # # 1) HDI+ over time
    # plt.figure()
    # for mode, df in dfs.items():
    #     plt.plot(df["day"], df["HDI_plus"], label=mode)
    # plt.xlabel("Day")
    # plt.ylabel("HDI+")
    # plt.title("HDI+ over time")
    # plt.legend()
    # hdi_path = "fig1_hdi_plus.png"
    # plt.savefig(hdi_path, dpi=200, bbox_inches="tight")
    # plt.close()
    
    # # 2) IEKV_7d over time
    # plt.figure()
    # for mode, df in dfs.items():
    #     plt.plot(df["day"], df["IEKV_7d"], label=mode)
    # plt.xlabel("Day")
    # plt.ylabel("IEKV smoothed")
    # plt.title("IEKV smoothed over time")
    # plt.legend()
    # iekv_path = "fig2_iekv_proxy.png"
    # plt.savefig(iekv_path, dpi=200, bbox_inches="tight")
    # plt.close()
    
    # # 3) Escalation share over time
    # plt.figure()
    # for mode, df in dfs.items():
    #     plt.plot(df["day"], df["E_share"], label=mode)
    # plt.xlabel("Day")
    # plt.ylabel("E-share (fraction of loops in escalation)")
    # plt.title("Escalation share over time")
    # plt.legend()
    # eshare_path = "fig3_escalation_share.png"
    # plt.savefig(eshare_path, dpi=200, bbox_inches="tight")
    # plt.close()
    
    # # Show quick summary table
    # summary = []
    # for mode, df in dfs.items():
    #     summary.append({
    #         "mode": mode,
    #         "avg_HDI_plus": float(df["HDI_plus"].mean()),
    #         "avg_IEKV_7d": float(df["IEKV_7d"].mean()),
    #         "avg_E_share": float(df["E_share"].mean()),
    #     })
    # summary_df = pd.DataFrame(summary)
    # summary_df  
    
    # # ---- сценарий scripted shocks (пример) ----
    # SCRIPTED_SHOCKS = {
    #     30: {"SUPPLY": True, "SANCTIONS_REGIME": 1, "SUPPLY_CHAIN_DELTA": -0.10, "SUPPLY_CHAIN_HALF_LIFE": 120},
    #     55: {"OUTBREAK": True, "OUTBREAK_SEV": 0.85},
    #     75: {"REVENUE": True},
    #     90: {"INFO": True},
    # }
    
    # # ---- прогон эксперимента ----
    # res = run_experiment(
    #     modes=("high_latency", "low_latency"),
    #     T_days=120,
    #     seed=42,
    #     scripted_shocks=SCRIPTED_SHOCKS,
    # )
    
    # # ---- подготовка рядов для Figure 4 ----
    # plt.figure(figsize=(9, 4.5))
    
    # for mode, logs in res.items():
    #     df = logs["state"].copy()
    
    #     # На случай, если нет явной колонки day
    #     if "day" not in df.columns:
    #         df["day"] = range(1, len(df) + 1)
    
    #     plt.plot(
    #         df["day"],
    #         df["HDI_plus"],
    #         label=mode,
    #         linewidth=2,
    #     )
    
    # # Вертикальные линии — дни шоков
    # for day in sorted(SCRIPTED_SHOCKS.keys()):
    #     plt.axvline(x=day, linestyle="--", linewidth=1, alpha=0.6)
    
    # plt.xlabel("Time (days)")
    # plt.ylabel("HDI⁺ ")
    # plt.title("HDI⁺ over time under scripted shock scenario")
    # plt.legend(frameon=False)
    # plt.grid(alpha=0.3)
    
    # plt.tight_layout()
    # plt.savefig("fig4_hdi_plus_scripted.png", dpi=300)
    # plt.show()
    
    # #=========================
    # #Monte-Carlo experiment
    # #=========================

    # MC_SEEDS = list(range(1, 51))   # 50-run Monte Carlo
    # T_DAYS = 120

    # SCRIPTED_SHOCKS = {
    #     30: {"SUPPLY": True, "SANCTIONS_REGIME": 1, "SUPPLY_CHAIN_DELTA": -0.10, "SUPPLY_CHAIN_HALF_LIFE": 120},
    #     55: {"OUTBREAK": True, "OUTBREAK_SEV": 0.85},
    #     75: {"REVENUE": True},
    #     90: {"INFO": True},
    # }

    # print("\n=== Monte-Carlo experiment ===")
    # print(f"Runs: {len(MC_SEEDS)}, Horizon: {T_DAYS} days")
    # print("Scenario: scripted compound shocks\n")

    # mc = run_monte_carlo(
    #     seeds=MC_SEEDS,
    #     modes=("high_latency", "low_latency"),
    #     T_days=T_DAYS,
    #     scripted_shocks=SCRIPTED_SHOCKS,
    # )

    # # Save raw results
    # mc.to_csv("monte_carlo_results.csv", index=False)
    # print("Saved: monte_carlo_results.csv")

    # # =========================
    # # Aggregated interpretation
    # # =========================

    # summary = (
    #     mc.groupby("mode")
    #       .agg(
    #           avg_HDI_plus_mean=("avg_HDI_plus", "mean"),
    #           avg_HDI_plus_std=("avg_HDI_plus", "std"),
    #           avg_IEKV_mean=("avg_IEKV_7d", "mean"),
    #           avg_IEKV_std=("avg_IEKV_7d", "std"),
    #           avg_E_share_mean=("avg_E_share", "mean"),
    #           avg_E_share_std=("avg_E_share", "std"),
    #           p90_IEKV_mean=("p90_IEKV_7d", "mean"),
    #           min_HDI_plus_mean=("min_HDI_plus", "mean"),
    #       )
    #       .reset_index()
    # )

    # print("\n=== Monte-Carlo summary (mean ± std) ===")
    # for _, r in summary.iterrows():
    #     print(
    #         f"\nMode: {r['mode']}\n"
    #         f"  avg HDI+      = {r['avg_HDI_plus_mean']:.3f}\n"
    #         f"  avg IEKV      = {r['avg_IEKV_mean']:.2f}\n"
    #         f"  avg E-share   = {r['avg_E_share_mean']:.2f}\n"
    #         f"  p90 IEKV      = {r['p90_IEKV_mean']:.2f}\n"
    #         f"  min HDI+      = {r['min_HDI_plus_mean']:.3f}"
    #     )

    # print("\nInterpretation:")
    # print(
    #     "- avg IEKV captures chronic managerial load\n"
    #     "- p90 IEKV captures tail risk of overload\n"
    #     "- min HDI+ captures welfare downside risk\n"
    #     "- Differences stable across random seeds ⇒ structural effect, not noise\n"
    # )
    
    # SEEDS = list(range(1, 51))  # N=50
    # T_DAYS = 120
    # MODES = ("high_latency", "low_latency")
    
    # SCRIPTED_SHOCKS = {
    #     30: {"SUPPLY": True, "SANCTIONS_REGIME": 1, "SUPPLY_CHAIN_DELTA": -0.10, "SUPPLY_CHAIN_HALF_LIFE": 120},
    #     55: {"OUTBREAK": True, "OUTBREAK_SEV": 0.85},
    #     75: {"REVENUE": True},
    #     90: {"INFO": True},
    # }
    
    # label_map = {
    #     "high_latency": "High-latency",
    #     "low_latency": "Low-latency",
    # }
    
    # # -------------------------
    # # Monte Carlo run (uses your existing helper)
    # # -------------------------
    # mc = run_monte_carlo(
    #     seeds=SEEDS,
    #     modes=MODES,
    #     T_days=T_DAYS,
    #     scripted_shocks=SCRIPTED_SHOCKS,   # set None for baseline MC without scripted shocks
    # )
    
    # mc.to_csv("mc_results.csv", index=False)
    # print("Saved:", "mc_results.csv")
    
    # # -------------------------
    # # Figure 5: Boxplots of average IEKV across runs
    # # -------------------------
    # plt.figure(figsize=(7.5, 4.5))
    
    # data_iekv = [
    #     mc.loc[mc["mode"] == "high_latency", "avg_IEKV_7d"].values,
    #     mc.loc[mc["mode"] == "low_latency", "avg_IEKV_7d"].values,
    # ]
    
    # plt.boxplot(
    #     data_iekv,
    #     labels=[label_map["high_latency"], label_map["low_latency"]],
    # )
    
    # plt.ylabel("Average IEKV (IEKV_smoothed)")
    # plt.title(f"Monte Carlo (N={len(SEEDS)}): Average IEKV by governance mode")
    # plt.grid(alpha=0.3)
    # plt.tight_layout()
    # plt.savefig("fig5_mc_box_avg_iekv.png", dpi=300)
    # plt.show()
    
    # # -------------------------
    # # Figure 6: Boxplots of minimum HDI+ across runs
    # # -------------------------
    # plt.figure(figsize=(7.5, 4.5))
    
    # data_min_hdi = [
    #     mc.loc[mc["mode"] == "high_latency", "min_HDI_plus"].values,
    #     mc.loc[mc["mode"] == "low_latency", "min_HDI_plus"].values,
    # ]
    
    # plt.boxplot(
    #     data_min_hdi,
    #     labels=[label_map["high_latency"], label_map["low_latency"]],
    # )
    
    # plt.ylabel("Minimum HDI⁺ (proxy)")
    # plt.title(f"Monte Carlo (N={len(SEEDS)}): Downside risk (min HDI⁺) by governance mode")
    # plt.grid(alpha=0.3)
    # plt.tight_layout()
    # plt.savefig("fig6_mc_box_min_hdi.png", dpi=300)
    # plt.show()
    
    # # Optional: tail overload risk
    # plt.figure(figsize=(7.5, 4.5))
    # data_p90 = [
    #     mc.loc[mc["mode"] == "high_latency", "p90_IEKV_7d"].values,
    #     mc.loc[mc["mode"] == "low_latency", "p90_IEKV_7d"].values,
    # ]
    # plt.boxplot(data_p90, labels=[label_map["high_latency"], label_map["low_latency"]])
    # plt.ylabel("p90 IEKV (IEKV_smoothed)")
    # plt.title(f"Monte Carlo (N={len(SEEDS)}): Tail overload risk (p90 IEKV) by governance mode")
    # plt.grid(alpha=0.3)
    # plt.tight_layout()
    # plt.savefig("figX_mc_box_p90_iekv.png", dpi=300)
    # plt.show()
    
    # 1 PDCA_PERIOD_HOURS
    # 2 NOOCRATIC_BASE_PERIOD_HOURS / NOOCRATIC_MIN_PERIOD_HOURS / NOOCRATIC_MAX_PERIOD_HOURS
    # 3 escalation_factor
    # 4 throughput_mult
    
    base = StateParams(seed=1, T_days=120)  # seed тут будет переопределён в sweep
    SEEDS = list(range(1, 51))

    SCRIPTED_SHOCKS = {
        30: {"SUPPLY": True, "SANCTIONS_REGIME": 1, "SUPPLY_CHAIN_DELTA": -0.10, "SUPPLY_CHAIN_HALF_LIFE": 120},
        55: {"OUTBREAK": True, "OUTBREAK_SEV": 0.85},
        75: {"REVENUE": True},
        90: {"INFO": True},
    }

    sens = run_sensitivity(
        base_params=base,
        param_name="throughput_mult",
        values=[0.75, 1.0, 1.25],
        seeds=SEEDS,
        modes=("high_latency", "low_latency"),
        T_days=120,
        scripted_shocks=SCRIPTED_SHOCKS,
    )

    sens.to_csv("sensitivity_throughput_mult.csv", index=False)

    plot_sensitivity_boxplots(
        sens,
        metric="avg_IEKV_7d",
        param_name="throughput_mult",
        title="Sensitivity: average IEKV (7d) vs throughput multiplier",
        ylabel="Average IEKV (7d)",
        save_path="fig_sens_avg_iekv_throughput.png",
    )
    