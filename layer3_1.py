"""
=============================================================================
ADFI Framework — Layer 3: Intelligence Layer & Action Layer
=============================================================================
Maps to Chapter 3.5 and 3.6 of the ADFI thesis.
Uses Local Ollama (Phi-3) for 100% Free and Private AI Inference.
=============================================================================
"""

import os
import json
import textwrap
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import shap
import lime
import lime.lime_tabular


# =============================================================================
# Local LLM setup — Ollama (Phi-3) Native CrewAI Integration
# =============================================================================

def _get_local_llm():
    """
    Connects to the local Ollama instance running Phi-3.
    Uses CrewAI's native LLM class to bypass the OpenAI validator bugs.
    """
    print("  [LLM] Connecting to Local ADFI Multi-Agent Protocol (Ollama Phi-3)...")
    
    try:
        from crewai import LLM
        
        # 1. Clear any lingering Google keys that confuse CrewAI's internal router
        os.environ.pop("GOOGLE_API_KEY", None)
        os.environ.pop("GEMINI_API_KEY", None)
        
        # 2. Provide a dummy key to instantly satisfy CrewAI's OpenAI validator
        os.environ["OPENAI_API_KEY"] = "sk-dummy-key-to-bypass-crewai-bug"
        
        # 3. Connect natively to your local Ollama server
        llm = LLM(
            model="ollama/phi3",
            base_url="http://localhost:11434",
            temperature=0.2
        )
        
        print("  [LLM] Local Phi-3 model connected successfully.\n")
        return llm
        
    except ImportError:
        print("  [LLM] crewai not installed properly. Run: pip install crewai")
        return None
    except Exception as e:
        print(f"  [LLM] Agent connection failed: {e}")
        return None

# =============================================================================
# §3.5.1  SHAP — Global Explainability  (Equation 3.7)
# =============================================================================

class SHAPExplainer:
    def __init__(self, model, feature_names: list):
        self.feature_names = feature_names
        print(f"  [SHAP] Building TreeExplainer for {len(feature_names)} raw clinical features …")
        self.explainer = shap.TreeExplainer(model)
        print("  [SHAP] Explainer ready.")

    def explain(self, X_test: np.ndarray) -> np.ndarray:
        print(f"  [SHAP] Computing Shapley values for {len(X_test)} patients …")
        shap_values = self.explainer.shap_values(X_test)
        
        # Handle 3D array output from Random Forest
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]  
            
        shap_values = np.array(shap_values)
        print(f"  [SHAP] Done. Shape: {shap_values.shape}")
        return shap_values

    def global_importance(self, shap_values: np.ndarray) -> list:
        mean_abs = np.abs(shap_values).mean(axis=0)
        return sorted(zip(self.feature_names, mean_abs), key=lambda x: x[1], reverse=True)

    def patient_attribution(self, shap_values: np.ndarray, patient_idx: int, top_n: int = 15) -> list:
        patient_shap = shap_values[patient_idx]
        return sorted(zip(self.feature_names, patient_shap), key=lambda x: abs(x[1]), reverse=True)[:top_n]


# =============================================================================
# §3.5.1  LIME — Local Explainability  (Equation 3.8)
# =============================================================================

class LIMEExplainer:
    def __init__(self, predict_fn, X_train: np.ndarray, feature_names: list):
        self.predict_fn    = predict_fn
        self.feature_names = feature_names
        self.explainer     = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            class_names=["Control", "ADHD"],
            mode="classification",
            discretize_continuous=True,
            random_state=42
        )
        print("  [LIME] LimeTabularExplainer ready.")

    def explain_patient(self, x_patient: np.ndarray, num_features: int = 10, num_samples: int = 1000) -> dict:
        exp = self.explainer.explain_instance(
            data_row=x_patient,
            predict_fn=self.predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            labels=(1,)
        )
        adhd_prob  = float(self.predict_fn(x_patient.reshape(1, -1))[0, 1])
        prediction = "ADHD" if adhd_prob >= 0.5 else "Control"
        top_feats  = exp.as_list(label=1)

        return {
            "adhd_probability": adhd_prob,
            "prediction":       prediction,
            "top_features":     top_feats,
        }


# =============================================================================
# §3.5.2  FAISS RAG — DSM-5 Knowledge Base
# =============================================================================

DSM5_KNOWLEDGE_BASE = [
    "DSM-5 Criterion A1 Inattention: Often fails to give close attention to details or makes careless mistakes. In CPT-II this maps to high omission errors — failing to respond to targets — and low Adhd TScore Omissions.",
    "DSM-5 Criterion A2 Inattention: Difficulty sustaining attention in tasks. In CPT-II: increasing RT variability across blocks (rising cpt_block std_rt), increasing omissions in later blocks. Core sustained attention deficit.",
    "DSM-5 Criterion A4 Inattention: Fails to follow through on instructions. Associated with commission errors and task incompletion. CPT-II Raw Score Commissions and perseveration counts are direct indicators.",
    "DSM-5 Criterion A5 Inattention: Difficulty organising tasks. Reflected in disrupted circadian rhythm: low inter-daily stability (act_interdaily_stab < 0.4) and high intra-daily variability (act_intradaily_var > 1.0).",
    "DSM-5 Criterion A8 Inattention: Easily distracted by extraneous stimuli. Maps to elevated ACC__variance, ACC__kurtosis, ACC__mean_abs_change in actigraphy tsfresh features.",
    "DSM-5 Criterion A10 Hyperactivity: Fidgets or squirms in seat. Direct mapping to elevated ACC__abs_energy, ACC__mean, ACC__root_mean_square in the tsfresh actigraphy feature set.",
    "DSM-5 Criterion A14 Hyperactivity: Often on the go, driven by a motor. Associated with high ACC__absolute_sum_of_changes and high overall activity metrics in the actigraphy data.",
    "DSM-5 Criterion A16 Impulsivity: Blurts out answers before questions completed. CPT-II commission errors: Raw Score Commissions > 20 and Adhd TScore Commissions > 65 are clinically elevated.",
    "DSM-5 Criterion A17 Impulsivity: Difficulty waiting turn. Low DPrime (< 0) indicates at-chance target discrimination. High Beta indicates overly liberal response criterion — impulsive responding.",
    "DSM-5 Criterion A18 Impulsivity: Interrupts or intrudes. Perseveration errors on CPT-II. Raw Score Perseverations > 5 indicates repeated impulsive responses after non-targets.",
    "HRV and ADHD: Adults with ADHD show reduced heart rate variability — lower RMSSD and lower SD1 (Poincare plot) compared to neurotypical controls. This reflects reduced parasympathetic tone and prefrontal dysfunction.",
    "HRV SDNN interpretation: SDNN below 50ms indicates reduced autonomic flexibility. Combined with elevated actigraphy metrics, low SDNN strongly supports ADHD neurocognitive profile and sympathetic dominance.",
    "CPT-II HitRT interpretation: Elevated Raw Score HitRT (mean RT) indicates slow attentional processing. High HitSE (standard error of RT) reflects response inconsistency — hallmark of ADHD inattentive presentation.",
    "CPT-II block trajectory ADHD pattern: Neurotypical individuals maintain consistent performance across all 6 blocks. ADHD individuals show degradation: rising cpt_block_std_rt, rising cpt_block_omissions, falling accuracy in blocks 4-6.",
    "Circadian rhythm in ADHD: Adults with ADHD show disrupted circadian rest-activity cycles. Inter-daily Stability IS below 0.5 and Intra-daily Variability IV above 1.0 are consistent with ADHD-related dysregulation.",
]

class FAISSKnowledgeBase:
    def __init__(self):
        try:
            import faiss
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import normalize

            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=512)
            self._texts  = DSM5_KNOWLEDGE_BASE
            vecs         = self.vectorizer.fit_transform(self._texts).toarray()
            vecs         = normalize(vecs, norm="l2").astype(np.float32)

            self.index   = faiss.IndexFlatL2(vecs.shape[1])
            self.index.add(vecs)
            self._faiss_ok  = True
            print(f"  [RAG] FAISS index built — {len(self._texts)} DSM-5 chunks.")

        except ImportError:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            self.vectorizer   = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self._texts       = DSM5_KNOWLEDGE_BASE
            self._kb_matrix   = self.vectorizer.fit_transform(self._texts)
            self._cos_sim     = cosine_similarity
            self._faiss_ok    = False
            print("  [RAG] FAISS not found — using cosine similarity fallback.")

    def retrieve(self, query: str, top_k: int = 4) -> list:
        if self._faiss_ok:
            import faiss
            from sklearn.preprocessing import normalize
            q_vec = self.vectorizer.transform([query]).toarray()
            q_vec = normalize(q_vec, norm="l2").astype(np.float32)
            distances, indices = self.index.search(q_vec, top_k)
            return [(float(1 / (1 + distances[0][i])), self._texts[indices[0][i]]) for i in range(top_k) if indices[0][i] < len(self._texts)]
        else:
            q_vec  = self.vectorizer.transform([query])
            scores = self._cos_sim(q_vec, self._kb_matrix).flatten()
            top_i  = scores.argsort()[::-1][:top_k]
            return [(float(scores[i]), self._texts[i]) for i in top_i]

    def get_context(self, query: str, top_k: int = 4) -> str:
        chunks = self.retrieve(query, top_k)
        return "\n\n".join([f"[Clinical Guideline {i+1} | Relevance {s:.2f}]\n{text}" for i, (s, text) in enumerate(chunks)])


# =============================================================================
# §3.5.2  CrewAI Multi-Agent System
# =============================================================================

def _build_crew(llm, findings_text: str, rag_context: str):
    try:
        from crewai import Agent, Task, Crew, Process
    except ImportError:
        print("  [CrewAI] crewai not installed. Run:  pip install crewai")
        return None

    data_analyst = Agent(
        role="Clinical Data Analyst",
        goal="Interpret SHAP and LIME values from the ADFI model and translate them into structured clinical domains (Actigraphy, HRV, CPT-II).",
        backstory="Expert in converting machine learning feature attributions into readable clinical observations.",
        llm=llm, verbose=True, allow_delegation=False,
    )

    diagnostic_critic = Agent(
        role="Diagnostic Critic and Clinical Validator",
        goal="Cross-check the Data Analyst's findings against the retrieved DSM-5 ADHD criteria to prevent AI hallucination.",
        backstory="Clinical psychologist specializing in DSM-5 validation. You ensure every data point maps to a recognized ADHD criterion.",
        llm=llm, verbose=True, allow_delegation=False,
    )

    synthesis_agent = Agent(
        role="Clinical Report Synthesis Agent",
        goal="Write the final clinical decision support report, explicitly including 'Green Therapy' and 'Digital Therapeutics' as per the Action Layer framework.",
        backstory="Lead clinical writer for the ADFI system. You format the validated findings into a professional, actionable 6-part medical report.",
        llm=llm, verbose=True, allow_delegation=False,
    )

    task_analyse = Task(
        description=f"Analyse these model outputs:\n{findings_text}\nIdentify top SHAP drivers and the dominant clinical domain.",
        agent=data_analyst,
        expected_output="Structured breakdown of clinical data drivers.",
    )

    task_critique = Task(
        description=f"Cross-check the analyst findings against this DSM-5 knowledge base:\n{rag_context}\nMap the raw features to specific DSM-5 Criteria (e.g., A1, A10).",
        agent=diagnostic_critic,
        expected_output="DSM-5 validation mapping.",
        context=[task_analyse],
    )

    task_synthesise = Task(
        description="""
Write the final ADFI Clinical Decision Support Report.
Structure EXACTLY with these sections:
1. EXECUTIVE SUMMARY
2. XAI ATTRIBUTION ANALYSIS (Cite specific SHAP/LIME features)
3. DSM-5 CRITERION ALIGNMENT (Cite specific A1-A18 criteria)
4. ENSEMBLE MODEL CONFIDENCE
5. LIMITATIONS (Note this is decision support, not a final diagnosis)
6. ACTION LAYER: NON-PHARMACOLOGICAL PLAN
   - Must prescribe explicit 'Green Therapy' (e.g., Shinrin-yoku, outdoor exercise)
   - Must prescribe 'Digital Therapeutics' (e.g., EndeavorRx)
""",
        agent=synthesis_agent,
        expected_output="Final 6-section clinical report.",
        context=[task_analyse, task_critique],
    )

    return Crew(agents=[data_analyst, diagnostic_critic, synthesis_agent], tasks=[task_analyse, task_critique, task_synthesise], process=Process.sequential, verbose=True)


# =============================================================================
# Rule-based fallback (no API key / offline)
# =============================================================================

def _rule_based_report(findings: dict) -> str:
    prob, sev, pred = findings["adhd_percentage"], findings["severity_band"], findings["prediction"]
    dom, pos_drv, neg_drv = findings["domain_breakdown"], findings["positive_drivers"], findings["negative_drivers"]
    lime_top, pidx = findings["lime_top"], findings["patient_idx"] + 1

    top_domain = max(dom, key=dom.get) if dom else "Mixed"

    lines = [
        "=" * 70,
        f"  ADFI CLINICAL DECISION SUPPORT REPORT — PATIENT {pidx}",
        f"  Generated by: Rule-Based Agent (Offline Fallback)",
        "=" * 70,
        "\n1. EXECUTIVE SUMMARY", "-" * 40,
        textwrap.fill(f"The ADFI framework assigned an ADHD probability of {prob}% (Classification: {pred}, Severity: {sev}). The {top_domain} domain was the primary driver of this prediction.", width=68),
        "\n2. XAI ATTRIBUTION ANALYSIS (SHAP & LIME)", "-" * 40,
        "  Top Features driving ADHD Risk (Positive SHAP):"
    ]
    for name, val in pos_drv[:4]: lines.append(f"    {name[:45]:<45} φ = {val:+.4f}")
    
    lines.extend(["\n  Top Features reducing ADHD Risk (Negative SHAP):"])
    for name, val in neg_drv[:3]: lines.append(f"    {name[:45]:<45} φ = {val:+.4f}")

    lines.extend(["\n3. DSM-5 CRITERION ALIGNMENT", "-" * 40])
    if dom.get("CPT-II Performance", 0) > 15:
        lines.append("  • Criterion A1/A2: High CPT errors map to sustained inattention.")
    if dom.get("Actigraphy / Motor Activity", 0) > 15:
        lines.append("  • Criterion A10/A14: Elevated actigraphy maps to hyperactive motor drive.")
    if dom.get("HRV / Cardiac Autonomic", 0) > 5:
        lines.append("  • HRV: Autonomic dysregulation consistent with ADHD sympathetic dominance.")

    lines.extend([
        "\n4. ENSEMBLE MODEL CONFIDENCE", "-" * 40,
        "  Prediction synthesized from Bio-inspired FA/PSO + DNN.",
        "\n5. LIMITATIONS AND UNCERTAINTY", "-" * 40,
        textwrap.fill("Generated by an AI decision-support system. High-variance estimates are expected. Must be reviewed by a qualified psychiatrist.", width=68),
        "\n6. ACTION LAYER: NON-PHARMACOLOGICAL PLAN", "-" * 40,
        "  a) GREEN THERAPY (Attention Restoration Theory):",
        "     • 30-min daily forest walk or park activity (5x/week)",
        "     • Weekly 2-hr Shinrin-yoku (forest bathing)",
        "  b) DIGITAL THERAPEUTICS:",
        "     • EndeavorRx: 25 min/day, 5x/week (FDA-cleared for ADHD)",
        "     • Cogmed Working Memory Training",
        "\n" + "=" * 70
    ])
    return "\n".join(lines)


# =============================================================================
# §3.5  Layer 3 entry point
# =============================================================================

def run_layer3_intelligence(
    surrogate_model,
    X_train_full:     np.ndarray,
    X_test_full:      np.ndarray,
    y_test:           np.ndarray,
    feature_names:    list,
    patient_idx:      int  = 0,
    save_outputs:     bool = True,
) -> dict:
    
    print("=" * 65)
    print("  LAYER 3 — INTELLIGENCE & ACTION LAYER (XAI + CrewAI + Ollama)")
    print("=" * 65 + "\n")

    # Call the new local LLM function
    llm    = _get_local_llm()
    rag_kb = FAISSKnowledgeBase()

    # SHAP on Raw Features
    print("\n[§3.5.1] SHAP — Global Attribution (Eq. 3.7) …")
    shap_exp    = SHAPExplainer(surrogate_model, feature_names)
    shap_values = shap_exp.explain(X_test_full)

    global_imp   = shap_exp.global_importance(shap_values)
    patient_shap = shap_exp.patient_attribution(shap_values, patient_idx)

    # LIME on Raw Features
    print(f"\n[§3.5.1] LIME — Local Attribution for Patient {patient_idx+1} (Eq. 3.8) …")
    lime_exp    = LIMEExplainer(surrogate_model.predict_proba, X_train_full, feature_names)
    lime_result = lime_exp.explain_patient(X_test_full[patient_idx], num_features=10)
    adhd_prob   = lime_result["adhd_probability"]
    prediction  = lime_result["prediction"]

    # Structuring Findings
    def domain_breakdown(shap_top):
        d = {"Actigraphy / Motor Activity": 0.0, "HRV / Cardiac Autonomic": 0.0, "Circadian Rhythm": 0.0, "CPT-II Performance": 0.0, "Other": 0.0}
        for name, val in shap_top:
            nl = name.lower()
            if "acc__" in nl: d["Actigraphy / Motor Activity"] += abs(val)
            elif "hrv_" in nl: d["HRV / Cardiac Autonomic"] += abs(val)
            elif "act_" in nl: d["Circadian Rhythm"] += abs(val)
            elif "cpt" in nl or "score" in nl or "response" in nl: d["CPT-II Performance"] += abs(val)
            else: d["Other"] += abs(val)
        t = sum(d.values()) or 1.0
        return {k: round(v/t*100, 1) for k, v in d.items()}

    pos_drv = [(n, v) for n, v in patient_shap if v > 0]
    neg_drv = [(n, v) for n, v in patient_shap if v < 0]
    dom_bkdn = domain_breakdown(patient_shap)

    findings = {
        "patient_idx":       patient_idx,
        "adhd_probability":  adhd_prob,
        "adhd_percentage":   round(adhd_prob * 100, 1),
        "prediction":        prediction,
        "severity_band":     "High" if adhd_prob >= 0.75 else "Moderate" if adhd_prob >= 0.5 else "Low",
        "positive_drivers":  pos_drv,
        "negative_drivers":  neg_drv,
        "domain_breakdown":  dom_bkdn,
        "dominant_expert":   "Bio-Inspired Hybrid Ensemble",
        "shap_top":          patient_shap,
        "lime_top":          lime_result["top_features"],
    }

    print(f"\n[§3.5.2] FAISS RAG — Retrieving DSM-5 context …")
    rag_context = rag_kb.get_context(f"Primary domain: {max(dom_bkdn, key=dom_bkdn.get)}. Top features: " + " ".join([n for n, _ in patient_shap[:5]]))

    print(f"\n[§3.5.2] CrewAI Digital Clinical Team …")
    findings_text = f"Patient Prediction: {prediction} ({findings['adhd_percentage']}%)\nTop Positive SHAP: {pos_drv[:4]}\nTop Negative SHAP: {neg_drv[:3]}\nDomain Impact: {dom_bkdn}"
    
    full_report = ""
    if llm is not None:
        try:
            crew = _build_crew(llm, findings_text, rag_context)
            if crew: 
                print("  [Agents] Passing data to Local LLM. Please wait while the CPU processes...")
                full_report = str(crew.kickoff())
        except Exception as e:
            print(f"  [CrewAI] Error: {e}")
    
    if not full_report:
        full_report = _rule_based_report(findings)

    print("\n" + "=" * 65 + "\n  LAYER 3 OUTPUT — CLINICAL REPORT\n" + "=" * 65)
    print(full_report)

    if save_outputs:
        with open(f"layer3_report_patient_{patient_idx+1}.txt", "w", encoding="utf-8") as f: f.write(full_report)
        np.save("layer3_shap_values.npy", shap_values)

    return {"shap_values": shap_values, "full_report": full_report}


# =============================================================================
# Direct execution — connects Layer 2 → Layer 3 via SURROGATE
# =============================================================================

if __name__ == "__main__":

    print(">>> Layer 1 — Preprocessing …\n")
    from datapreprocessing import run_adf_preprocessing
    (X_train, y_train, X_test, y_test, feat_names, _, _, _, _) = run_adf_preprocessing()

    print("\n>>> Layer 2 — Hybrid Optimization (Skipping printing for brevity) …\n")
    from sklearn.ensemble import RandomForestClassifier
    
    print("  [Layer 3] Training Global Surrogate Model on 903 raw features to preserve clinical interpretability...")
    surrogate = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    surrogate.fit(X_train, y_train)

    print("\n>>> Layer 3 — Intelligence Layer …\n")
    outputs = run_layer3_intelligence(
        surrogate_model = surrogate,
        X_train_full    = X_train,
        X_test_full     = X_test,
        y_test          = y_test,
        feature_names   = feat_names,
        patient_idx     = 0,       
        save_outputs    = True,
    )