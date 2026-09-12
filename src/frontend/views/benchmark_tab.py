"""
Academic Evaluation & Benchmark Suite View (Tab 6).
Executes live empirical evaluation benchmarks across Scenarios A, B, C, and D.
"""
import streamlit as st
import pandas as pd
from src.backend.benchmark_suite import ExperimentSuite


def render_benchmark_tab():
    st.subheader("Academic Review & Empirical Evaluation Suite")
    st.caption("Direct, empirical reproduction of Scenarios A, B, C, and D from Section V of the Project Review Report.")

    if st.button("🚀 Run Live Experimental Benchmarks", type="primary", use_container_width=True):
        with st.spinner("Executing empirical experiments across vision, search, allocation, and triage modules..."):
            res_a = ExperimentSuite.run_scenario_a_vision()
            res_b = ExperimentSuite.run_scenario_b_routing()
            res_c = ExperimentSuite.run_scenario_c_allocation()
            res_d = ExperimentSuite.run_scenario_d_medical()

            st.session_state.bench_results = {
                "a": res_a,
                "b": res_b,
                "c": res_c,
                "d": res_d
            }
        st.success("All empirical benchmarks evaluated successfully.")

    if "bench_results" in st.session_state:
        br = st.session_state.bench_results

        # Scenario A
        st.markdown("#### Scenario A: Visual Damage & Hazard Assessment Evaluation")
        st.caption("Precision, Recall, and F1 on local test images:")
        st.dataframe(pd.DataFrame(br["a"]["table"]), hide_index=True, use_container_width=True)
        m_a = br["a"]["metrics"]
        st.markdown(f"**Empirical Metrics:** Precision: `{m_a['Precision']}` | Recall (Sensitivity): `{m_a['Recall (Sensitivity)']}` | F1-Score: `{m_a['F1 Score']}`")

        st.markdown("---")
        # Scenario B
        st.markdown("#### Scenario B: Route Planning Search Algorithm Comparison")
        st.caption("Comparing A* against Dijkstra (UCS), BFS, and Greedy Best-First Search on the disaster graph:")
        st.dataframe(pd.DataFrame(br["b"]["summary_table"]), hide_index=True, use_container_width=True)

        st.markdown("---")
        # Scenario C
        st.markdown("#### Scenario C: Resource Allocation Efficiency")
        st.caption("Priority-aware resource assignment compared against unprioritized nearest-available baseline:")
        st.dataframe(pd.DataFrame(br["c"]["table"]), hide_index=True, use_container_width=True)
        m_c = br["c"]["metrics"]
        st.markdown(f"**Empirical Results:** High-Priority ALS Coverage: `{m_c['High-Priority ALS Coverage Rate']}` | Trauma Beds Preserved for Non-Critical: `{m_c['Level-1 Trauma Beds Preserved for Non-Critical']}`")

        st.markdown("---")
        # Scenario D
        st.markdown("#### Scenario D: Medical Decision Support Comparison")
        st.caption("Comparing Rule-Only, Bayesian-Only, and Hybrid Triage against expert-authored benchmark cases:")
        st.dataframe(pd.DataFrame(br["d"]["table"]), hide_index=True, use_container_width=True)
        m_d = br["d"]["metrics"]
        st.markdown(f"""
        - **Rule-Only Accuracy:** `{m_d['Rule-Only Accuracy']:.1%}`
        - **Bayesian-Only Accuracy:** `{m_d['Bayesian-Only Accuracy']:.1%}`
        - **Hybrid Arbitrator Accuracy:** `{m_d['Hybrid Arbitrator Accuracy']:.1%}`
        - **High-Priority (RED) Sensitivity:** `{m_d['High-Priority Sensitivity (RED)']:.1%}`
        - **Critical False Negatives:** `{m_d['Critical False Negatives']}` (Conservative escalation prevents unsafe under-triage)
        """)
    else:
        st.info("Click 'Run Live Experimental Benchmarks' above to execute all 4 experimental evaluations live on your local environment.")
