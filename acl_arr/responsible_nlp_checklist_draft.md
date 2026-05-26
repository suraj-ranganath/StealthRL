# Responsible NLP Checklist Draft for ARR Submission

This is a working draft for the ARR submission form. Verify all answers against the final paper and your actual research/writing process before submission.

## A. Every Submission

**A1. Limitations described?** Yes.
Relevant sections: Section 7, "Limitations"; Section 6, "Conclusion and Future Work".

**A2. Potential risks discussed?** Yes.
Relevant sections: Section 8, "Ethical Considerations"; Section 7, "Limitations".
The paper discusses dual-use risk, detector-evasion misuse, and frames StealthRL as a robustness stress-testing tool.

## B. Scientific Artifacts

**B1. Creators of artifacts cited?** Yes.
Relevant sections: Section 2, "Related Work"; Section 4.1, "Dataset"; Section 4.2, "Detectors"; references.
The paper cites MAGE, RoBERTa/OpenAI detector, Fast-DetectGPT, Binoculars, MAGE detector, Qwen/Qwen3, LoRA, GRPO, E5, BERTScore, DeBERTa, AuthorMist, SilverSpeak, and related baselines.

**B2. Licenses or terms discussed?** Partially.
Relevant sections: Section 4.1, Section 4.2, Appendix I.
The paper identifies the artifacts and checkpoints used. If ARR requires explicit license names for every artifact, add a short license note or provide license details in the checklist answer.

**B3. Use consistent with intended use?** Yes.
Relevant sections: Section 8, "Ethical Considerations"; Section 4, "Experimental Setup".
Artifacts are used for research benchmarking, robustness evaluation, and detector stress-testing rather than production deployment.

**B4. PII/offensive-content checks discussed?** No / not applicable.
Justification: The work uses the existing MAGE benchmark and does not collect new user data or infer protected attributes. The paper applies token-length filtering and evaluates benchmark text only.

**B5. Artifact documentation provided?** Yes.
Relevant sections: Section 4.1, "Dataset"; Section 4.2, "Detectors"; Appendix I, "Dataset Statistics"; Appendix E, "Hyperparameters and Configuration".

**B6. Dataset statistics reported?** Yes.
Relevant sections: Section 4.1, "Dataset"; Appendix I, "Dataset Statistics".

## C. Computational Experiments

**C1. Model parameters / compute budget / runtime reported?** Yes.
Relevant sections: Section 4.4, "Implementation Details"; Appendix E, "Hyperparameters and Configuration".

**C2. Experimental setup and hyperparameters reported?** Yes.
Relevant sections: Section 4.4; Appendix E; Appendix F, "Model-Selection Ablations".

**C3. Descriptive statistics / variance reported?** Yes.
Relevant sections: Section 4.5, "Evaluation Metrics"; Section 5.2, "Robustness, Transfer, and Quality"; Appendix A; Appendix G.
The paper reports bootstrap confidence intervals and stochastic repeat analysis.

**C4. Package/model implementations and settings reported?** Yes.
Relevant sections: Section 4.2, "Detectors"; Section 4.4; Appendix E; Appendix H, "LLM Judge Prompt Templates".

## D. Human Participants / Annotators

**D1-D5. Human participants or annotators used?** No.
Justification: The paper does not recruit human annotators or run human-subjects data collection. Quality evaluation uses an automated LLM judge, and the exact prompts are reported in Appendix H.

## E. AI Assistants

**E1. AI assistants used?** Authors should answer truthfully based on the final research and writing workflow.
Suggested response if applicable: "Yes. AI assistants were used for coding support, LaTeX formatting, and/or language-editing assistance under author supervision. All scientific claims, experiments, results, and final text were reviewed and approved by the authors. See Sections 4--8 and appendices for experimental details and reproducibility information."
