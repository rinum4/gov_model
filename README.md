# Governance Under Continuous Disturbance – Simulation Model

This repository contains the simulation model used in the article:

“Governability Under Continuous Disturbance: A System Dynamics Perspective”
(submitted to *Systems Research and Behavioral Science*).

## Overview

The model implements a minimal system-dynamics simulation of governance systems operating under continuous low-intensity disturbance.  
It focuses on decision-processing capacity, escalation dynamics, and governance load accumulation under different architectural regimes.

Two governance regimes are compared:
- **High-latency regime** (hierarchical, delayed feedback)
- **Low-latency regime** (adaptive, accelerated review cycles)

## Key Outputs

The simulation generates:
- HDI⁺ (welfare proxy)
- IEKV (Integrated Governance Load Index)
- Escalation share across governance loops
- Monte Carlo statistics across multiple random seeds

## How to Run

```bash
python wp_state_experiment_v1.py
