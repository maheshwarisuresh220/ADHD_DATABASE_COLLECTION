"""
=============================================================================
ADFI Framework — Layer 3: Intelligence Layer (Local Secure simulation)
=============================================================================
Maps to Chapter 3.5 and 3.6 of the ADFI thesis.
Uses GPT4All for local inference (no API keys) and LangGraph for agent orchestration.
=============================================================================
"""

import os
import textwrap
import warnings
from typing import TypedDict
import numpy as np
import shap
import lime
import lime.lime_tabular
from gpt4all import GPT4All
from langgraph.graph import StateGraph, END

warnings.filterwarnings("ignore")

# =============================================================================
# §3.5.2 Local Secure LLM Wrapper (GPT4All)
# =============================================================================

class SecureLocalLLM:
    """Wrapper to make GPT4All work within LangGraph nodes."""
    def __init__(self, model_name="Phi-3-mini-4k-instruct.Q4_0.gguf"):
        print(f"  [LLM] Loading Local Secure Model ({model_name})...")
        # Downloads to ~/.cache/gpt4all/ on first run (~2.2GB)
        self.model = GPT4All(model_name)

    def invoke(self, prompt: str) -> str:
        """Synchronous generation for local agent processing."""
        with self.model.chat_session():
            return self.model.generate(prompt, max_tokens=1024)

# Global initialization to keep the model in RAM across agent nodes
llm_engine = SecureLocalLLM()

# =============================================================================
# §3.5.1 Explainable AI Logic (SHAP & LIME)
# =============================================================================

class SHAPExplainer:
    def __init__(self, model, feature_names: list):
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)

    def explain(self, X_test: np.ndarray) -> np.ndarray:
        shap_values = self.explainer.shap_values(X_test)
        # Slicing logic for ADHD class (index 1) in 3D Random Forest output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
        return np.array(shap_values)

    def patient_attribution(self, shap_values: np.ndarray, patient_idx: int, top_n: int = 10) -> list:
        patient_shap = shap_values[patient_idx]
        return sorted(zip(self.feature_names, patient_shap), key=lambda x: abs(x[1]), reverse=True)[:top_n]

class LIMEExplainer:
    def __init__(self, predict_fn, X_train: np.ndarray, feature_names: list):
        self.predict_fn = predict_fn
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train, feature_names=feature_names,
            class_names=["Control", "ADHD"], mode="classification", random_state=42
        )

    def explain_patient(self, x_patient: np.ndarray) -> dict:
        prob = float(self.predict_fn(x_patient.reshape(1, -1))[0, 1])
        return {"adhd_probability": prob, "prediction": "ADHD" if prob >= 0.5 else "Control"}

# =============================================================================
# §3.5.2 Agentic Workflow (LangGraph Orchestration)
# =============================================================================

class AgentState(TypedDict):
    findings: str
    analysis: str
    validation: str
    final_report: str

def clinical_analyst_node(state: AgentState):
    """Node 1: Interprets numerical XAI data into clinical domains."""
    prompt = f"As a Clinical Analyst, explain these feature attributions as medical symptoms:\n{state['findings']}"
    return {"analysis": llm_engine.invoke(prompt)}

def diagnostic_critic_node(state: AgentState):
    """Node 2: Validates analysis against standard DSM-5 ADHD criteria."""
    prompt = f"As a Critic, verify if this clinical analysis matches DSM-5 ADHD criteria:\n{state['analysis']}"
    return {"validation": llm_engine.invoke(prompt)}

def report_synthesis_node(state: AgentState):
    """Node 3: Generates final report and prescribes ART / Green Therapy."""
    prompt = f"Generate a formal medical report and prescribe 'Green Therapy' based on this validation:\n{state['validation']}"
    return {"final_report": llm_engine.invoke(prompt)}

def run_agentic_workflow(findings_text: str):
    """Builds and executes the sequential multi-agent graph."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("analyst", clinical_analyst_node)
    workflow.add_node("critic", diagnostic_critic_node)
    workflow.add_node("synthesis", report_synthesis_node)
    
    workflow.set_entry_point("analyst")
    workflow.add_edge("analyst", "critic")
    workflow.add_edge("critic", "synthesis")
    workflow.add_edge("synthesis", END)
    
    app = workflow.compile()
    print("  [Agents] Local multi-agent coordination active...")
    return app.invoke({"findings": findings_text})["final_report"]

# =============================================================================
# Main Execution Logic
# =============================================================================

def run_layer3_intelligence(surrogate, X_train, X_test, feat_names, patient_idx=0):
    print("=" * 65)
    print("  LAYER 3 — INTELLIGENCE LAYER (SECURE LOCAL SIMULATION)")
    print("=" * 65 + "\n")

    # 1. Generate XAI explanations using the Global Surrogate
    shap_exp = SHAPExplainer(surrogate, feat_names)
    vals = shap_exp.explain(X_test)
    top_feats = shap_exp.patient_attribution(vals, patient_idx)
    
    lime_exp = LIMEExplainer(surrogate.predict_proba, X_train, feat_names)
    l_res = lime_exp.explain_patient(X_test[patient_idx])

    # 2. Package data for the local digital clinical team
    findings_text = f"Initial Prediction: {l_res['prediction']} ({l_res['adhd_probability']*100:.1f}%)\n"
    findings_text += f"Primary Clinical Drivers: {top_feats}"

    # 3. Execute Agent Workflow
    report = run_agentic_workflow(findings_text)

    print("\n" + "="*65 + "\n FINAL CLINICAL REPORT (LOCAL GENERATION)\n" + "="*65)
    print(report)

if __name__ == "__main__":
    from datapreprocessing import run_adf_preprocessing
    from sklearn.ensemble import RandomForestClassifier

    # Load Data (using previous Layer 1 results)
    (X_train, y_train, X_test, y_test, feat_names, _, _, _, _) = run_adf_preprocessing()

    # Train a Local Surrogate model on raw features for interpretability
    # This prevents the agents from seeing "PCA Components" and lets them see clinical labels
    print("\n  [Surrogate] Training Random Forest on 903 raw features...")
    surrogate = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42).fit(X_train, y_train)

    run_layer3_intelligence(surrogate, X_train, X_test, feat_names, patient_idx=0)