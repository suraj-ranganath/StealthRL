"""
StealthBench: Unified evaluation harness for AI text detectors.
"""

from typing import List, Dict, Optional
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from ..detectors import FastDetectGPTDetector, GhostbusterDetector, BinocularsDetector
from .metrics import compute_auroc, compute_fpr_at_tpr, compute_bertscore, compute_fpr_gap


class StealthBench:
    """
    Unified evaluation harness for comparing multiple AI text detectors.
    
    StealthBench runs multiple detectors on common text sets and produces
    standardized metrics and comparison plots.
    """
    
    def __init__(
        self,
        detectors: List[str],
        output_dir: str = "outputs/stealthbench_results",
        device: str = "cuda"
    ):
        """
        Initialize StealthBench.
        
        Args:
            detectors: List of detector names to evaluate
            output_dir: Directory for output files
            device: Device to run detectors on
        """
        self.detector_names = detectors
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.detector_instances = {}
        self._initialize_detectors()
        
    def _initialize_detectors(self):
        """Initialize detector instances."""
        detector_map = {
            "fast-detectgpt": FastDetectGPTDetector,
            "ghostbuster": GhostbusterDetector,
            "binoculars": BinocularsDetector,
        }
        
        for detector_name in self.detector_names:
            if detector_name in detector_map:
                self.detector_instances[detector_name] = detector_map[detector_name](device=self.device)
            else:
                print(f"Warning: Unknown detector {detector_name}")
        
    def run(
        self,
        human_texts: List[str],
        ai_texts: List[str],
        paraphrased_texts: Optional[List[str]] = None,
        esl_texts: Optional[List[str]] = None,
        native_texts: Optional[List[str]] = None,
        esl_labels: Optional[List[int]] = None,
        native_labels: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Run StealthBench evaluation.
        
        Args:
            human_texts: Human-written texts
            ai_texts: AI-generated texts
            paraphrased_texts: StealthRL-paraphrased texts (optional)
            esl_texts: ESL writing samples for fairness analysis (optional)
            native_texts: Native writing samples for fairness analysis (optional)
            esl_labels: Labels for ESL texts (optional)
            native_labels: Labels for native texts (optional)
            
        Returns:
            DataFrame with evaluation results
        """
        print("Running StealthBench evaluation...")
        
        results = []
        
        # Prepare labels
        y_true = [0] * len(human_texts) + [1] * len(ai_texts)
        all_texts = human_texts + ai_texts
        
        # Run each detector
        for detector_name, detector in self.detector_instances.items():
            print(f"\nEvaluating {detector_name}...")
            
            try:
                # Get detection scores
                scores = detector.detect(all_texts)
                scores_list = scores.cpu().numpy().tolist()
                
                # Compute metrics
                auroc = compute_auroc(y_true, scores_list)
                fpr_05 = compute_fpr_at_tpr(y_true, scores_list, target_tpr=0.995)  # FPR@0.5% TPR
                fpr_1 = compute_fpr_at_tpr(y_true, scores_list, target_tpr=0.99)    # FPR@1% TPR
                
                # Compute fairness metrics if provided
                esl_fpr_gap = 0.0
                if esl_texts and native_texts and esl_labels and native_labels:
                    esl_scores = detector.detect(esl_texts).cpu().numpy().tolist()
                    native_scores = detector.detect(native_texts).cpu().numpy().tolist()
                    esl_fpr_gap = compute_fpr_gap(
                        esl_labels, esl_scores,
                        native_labels, native_scores,
                        threshold=0.5
                    )
                
                result = {
                    "detector": detector_name,
                    "auroc": auroc,
                    "fpr_at_0.5_tpr": fpr_05,
                    "fpr_at_1_tpr": fpr_1,
                    "esl_fpr_gap": esl_fpr_gap,
                }
                
                # Evaluate paraphrased texts if provided
                if paraphrased_texts:
                    paraph_scores = detector.detect(paraphrased_texts)
                    result["mean_paraphrased_score"] = float(paraph_scores.mean())
                    
                    # Compute BERTScore if we have original AI texts
                    if len(paraphrased_texts) == len(ai_texts):
                        bert_scores = compute_bertscore(
                            ai_texts[:len(paraphrased_texts)],
                            paraphrased_texts,
                            verbose=False
                        )
                        result["bertscore_f1"] = bert_scores["f1"]
                
                results.append(result)
                print(f"  AUROC: {auroc:.4f}, FPR@0.5%: {fpr_05:.4f}, FPR@1%: {fpr_1:.4f}")
                
            except Exception as e:
                print(f"  Error evaluating {detector_name}: {e}")
                continue
        
        df = pd.DataFrame(results)
        return df
        
    def save_results(self, results: pd.DataFrame, filename: str = "results.csv"):
        """Save evaluation results to CSV."""
        output_path = self.output_dir / filename
        results.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
        
    def generate_plots(self, results: pd.DataFrame):
        """Generate comparison plots."""
        if results.empty:
            print("No results to plot")
            return
        
        # Set style
        sns.set_style("whitegrid")
        
        # Plot AUROC comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # AUROC
        ax = axes[0]
        sns.barplot(data=results, x="detector", y="auroc", ax=ax)
        ax.set_title("AUROC Comparison")
        ax.set_ylabel("AUROC")
        ax.set_xlabel("Detector")
        ax.set_ylim([0, 1])
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # FPR@0.5%
        ax = axes[1]
        sns.barplot(data=results, x="detector", y="fpr_at_0.5_tpr", ax=ax)
        ax.set_title("FPR @ 99.5% TPR")
        ax.set_ylabel("FPR")
        ax.set_xlabel("Detector")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # ESL FPR Gap
        ax = axes[2]
        if "esl_fpr_gap" in results.columns:
            sns.barplot(data=results, x="detector", y="esl_fpr_gap", ax=ax)
            ax.set_title("ESL vs Native FPR Gap")
            ax.set_ylabel("FPR Gap")
            ax.set_xlabel("Detector")
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plot_path = self.output_dir / "detector_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to {plot_path}")
        plt.close()
