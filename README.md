# RescueMedAI 🚑
**AI-Powered Disaster Response and Emergency Medical Decision Support System**

> **Course:** Artificial Intelligence  
> **Faculty:** Professor Hemalatha S  
> **Team:** 
> - Keshav Kohli (24BIT0619)
> - Bharath D Singh (24BIT0458)
> - R. Rahul Padmanaban (24BIT0473)
>
> **Safety Notice:** *AI-Assisted Emergency Decision Support — Prototype — Human Clinical Review Required.*  
> *This system provides educational decision support and does not provide clinical diagnosis, autonomous medical triage, or validated therapeutic directives.*

---

## 🌟 Overview

During disasters such as earthquakes, flash floods, and urban fires, traditional navigation maps fail because roads are blocked, bridges are damaged, and hospital capacities fluctuate. Furthermore, field responders must triage victims under extreme uncertainty without delaying hospital preparations.

**RescueMedAI** solves this through a **Single Incident State** pipeline linking 9 stages:
1. **Disaster Image Ingestion:** Local imagery (drone/CCTV/mobile) with spatial hazard annotations.
2. **Visual Damage & Obstruction Analysis:** Real local computer vision (rubble gradient edge analysis, fire/smoke hue masks) and benchmark xBD-style ground truth tags.
3. **Disaster Severity Estimation:** Transparent, explainable weighted severity scoring ($Low, Medium, High, Critical$).
4. **Dynamic Hazard-Weighted Road Graph:** Road network with dynamic edge penalties and closures.
5. **A\* Route Planning:** $f(n) = g(n) + h(n)$ with an admissible Euclidean travel-time heuristic, benchmarked against Dijkstra (UCS), BFS, and Greedy Best-First Search.
6. **Priority-Aware Resource Allocation:** Multi-criteria matching of ALS/BLS ambulances, rescue teams, and receiving trauma centers with capacity preservation.
7. **Victim Vitals Ingestion:** Rapid responder field entry of physiological indicators and trauma symptoms.
8. **Hybrid Medical Decision Support:** Explicit START/ESI rule engine combined with a discrete Bayesian Network (CPTs + Shannon entropy) featuring conservative safety escalation.
9. **Hospital Pre-Arrival SBAR Dispatch:** Automated SBAR/MIST emergency handoff transmission for receiving emergency departments.

---

## 🚀 Key Implementation Features

- **100% Offline & Local:** Runs without internet access, zero external APIs, and no Google Maps API keys.
- **Pure Algorithmic Engines:** Exact implementations of A*, Dijkstra, BFS, Greedy Search, and discrete Bayesian inference in pure Python and NumPy.
- **Empirical Academic Evaluation:** Live execution of the 4 experimental scenarios from page 8 of the project review report without fabricated metrics.
- **Mission Control UI:** Streamlit-powered emergency operations command center with dark theme, interactive tactical graph plots, and glowing triage status badges.

---

## 📁 Repository Structure

```text
RescueMed/
├── app.py                          # Streamlit application entry point
├── requirements.txt                # Python package dependencies
├── README.md                       # Documentation & viva presentation guide
│
├── data/
│   ├── sample_images/              # Offline aerial disaster imagery
│   ├── sample_scenarios.json       # Predefined deterministic disaster scenarios
│   └── sample_patients.json        # Benchmark patient cases for Scenario D evaluation
│
├── src/
│   ├── state.py                    # Unified IncidentState contract
│   ├── benchmark_suite.py          # Empirical reproduction of Scenarios A, B, C, D
│   │
│   ├── routing/
│   │   ├── graph.py                # RoadNetwork with dynamic hazard penalties & closures
│   │   ├── search.py               # A*, Dijkstra/UCS, BFS, Greedy Best-First Search
│   │   └── benchmark.py            # Comparative search engine
│   │
│   ├── triage/
│   │   ├── rules.py                # Deterministic START/ESI emergency rule engine
│   │   ├── bayesian.py             # Exact discrete Bayesian Network in pure NumPy
│   │   └── arbitrator.py           # Hybrid arbitrator with conservative safety escalation
│   │
│   ├── allocation/
│   │   └── allocator.py            # Priority-aware matching vs nearest baseline
│   │
│   ├── vision/
│   │   ├── detector.py             # Local CV gradient/color filters & benchmark annotations
│   │   └── severity.py             # Multi-factor transparent severity estimator
│   │
│   ├── hospital/
│   │   └── notification.py         # Standardized SBAR/MIST emergency handoff generator
│   │
│   └── ui/
│       └── components.py           # Dark-theme styling and Matplotlib tactical network map
│
└── tests/
    ├── test_routing.py             # Tests for A* optimality, blockage avoidance, graph search
    ├── test_triage.py              # Tests for rule triggers, Bayesian inference, and safety
    ├── test_allocation.py          # Tests for priority scoring and baseline comparison
    ├── test_vision.py              # Tests for local pixel analysis and severity calculation
    └── test_benchmarks.py          # Tests for Scenarios A, B, C, and D evaluation suite
```

---

## 🛠️ Quick Start Guide

### 1. Launch the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 2. Run Automated Test Suite
To verify all algorithmic components:
```bash
pytest tests/ -v
```
All 23 unit and integration tests will run and pass in under 1 second.

---

## 🎯 Viva & Demonstration Walkthrough

When presenting RescueMedAI to faculty or reviewers:

1. **Sidebar Scenarios:** Switch between *Scenario 1 (Earthquake Collapse)*, *Scenario 2 (Flash Flood)*, and *Scenario 3 (Urban Commercial Fire)*.
2. **Tab 1 (Perception):** Explain the visual damage overlay, detection source transparency, and the weighted formula breakdown calculating disaster severity.
3. **Tab 2 (Road Graph & A\*):** 
   - Point out the tactical road network map.
   - Use the **Interactive Road Blockage Simulator** expander to block a road and show live dynamic A* rerouting.
   - Review the **Search Comparison Table**: demonstrate that A* and Dijkstra achieve the exact same optimal cost while observing node expansions and runtimes.
4. **Tab 3 (Resource Allocation):** Highlight how Red-priority victims are dispatched ALS ambulances and routed to Level-1 Trauma centers while Green-priority victims are routed to Community Care facilities to protect ICU capacity.
5. **Tab 4 (Field Medical Triage):** 
   - Modify a patient's vitals (e.g. drop SpO2 to 84% or BP to 75).
   - Click **Recompute Triage** and observe the live Bayesian posterior bar chart update instantly alongside the activated deterministic safety rules.
   - Note the **Conservative Safety Escalation** audit trail and mandatory safety disclaimer.
6. **Tab 5 (Hospital SBAR Console):** Display the formatted SBAR dispatch report and facility preparedness directives.
7. **Tab 6 (Academic Review & Benchmarks):** Click **Run Live Experimental Benchmarks** to execute all 4 experimental scenarios from the report live and display empirical tables and metrics.
