"""
StealthRL Evaluation Pipeline.

This module provides a comprehensive evaluation suite for StealthRL,
including dataset loading, multiple detectors, attack methods, and metrics.

Designed to be decoupled from training for paper-ready evaluation.

Usage:
    python -m eval.run --datasets mage --methods m0 m1 m2 --detectors roberta fast_detectgpt

Enhanced runner with structured output:
    python -m eval.runner_enhanced --name experiment_v1 --datasets mage raid
"""

from .data import (
    MAGEDataset,
    RAIDDataset,
    PadBenDataset,
    BaseEvalDataset,
    EvalSample,
    load_eval_dataset,
    prepare_mage_eval,
    prepare_raid_slice,
)
from .detectors import (
    BaseEvalDetector,
    RoBERTaOpenAIDetector,
    FastDetectGPTDetector,
    DetectGPTDetector,
    BinocularsDetector,
    GhostbusterDetector,
    EnsembleDetector,
    get_detector,
    load_detectors,
    DETECTOR_REGISTRY,
    DETECTOR_CONVENTIONS,
)
from .methods import (
    BaseAttackMethod,
    AttackOutput,
    NoAttack,
    SimpleParaphrase,
    StealthRLAttack,
    AdversarialParaphrasing,
    AuthorMist,
    HomoglyphAttack,
    HomoglyphSweep,
    sanitize_text,
    get_method,
    METHOD_REGISTRY,
    GUIDANCE_VARIANTS,
)
from .metrics import (
    compute_auroc,
    compute_tpr_at_fpr,
    compute_asr,
    compute_bootstrap_ci,
    compute_detector_metrics,
    compute_quality_metrics,
    calibrate_thresholds,
    DetectorMetrics,
    QualityMetrics,
    E5SimilarityScorer,
    PerplexityScorer,
)
from .plots import (
    # Constants
    COLORBLIND_COLORS,
    METHOD_NAMES,
    COLORS,
    # Original plots
    create_heatmap,
    create_tradeoff_plot,
    create_budget_sweep_plot,
    # AuthorMist-style plots
    create_auroc_bar_chart,
    create_auroc_radar_chart,
    create_method_comparison_summary,
    # Adversarial Paraphrasing-style plots
    create_roc_curves,
    create_roc_curves_logscale,
    create_transferability_heatmap,
    create_quality_likert_chart,
    create_winrate_chart,
    create_perplexity_comparison_table,
    create_combined_results_table,
    # Score visualization plots
    create_score_distribution_plot,
    create_score_shift_plot,
    create_human_ai_separation_plot,
    # Tables
    create_main_results_table,
    create_transfer_table,
    create_quality_table,
    generate_all_plots,
    generate_all_tables,
    # GPT-4o quality evaluation helpers
    QUALITY_RATING_PROMPT,
    WIN_RATE_PROMPT,
    get_quality_rating_messages,
    get_win_rate_messages,
    parse_quality_rating_response,
    parse_win_rate_response,
)
from .runner import EvalRunner

# Import sanitization module
from .sanitize import (
    sanitize,
    compute_sanitization_diff,
    run_sanitize_evaluation,
    create_sanitize_report,
    SanitizationResult,
    ZERO_WIDTH_CHARS,
    HOMOGLYPH_MAP,
)

# Import enhanced runner if available
try:
    from .runner_enhanced import (
        RunManager,
        EnhancedEvalRunner,
    )
except ImportError:
    RunManager = None
    EnhancedEvalRunner = None

__all__ = [
    # Data
    "MAGEDataset",
    "RAIDDataset",
    "PadBenDataset",
    "BaseEvalDataset",
    "EvalSample",
    "load_eval_dataset",
    "prepare_mage_eval",
    "prepare_raid_slice",
    # Detectors
    "BaseEvalDetector",
    "RoBERTaOpenAIDetector",
    "FastDetectGPTDetector",
    "DetectGPTDetector",
    "BinocularsDetector",
    "GhostbusterDetector",
    "EnsembleDetector",
    "get_detector",
    "load_detectors",
    "DETECTOR_REGISTRY",
    "DETECTOR_CONVENTIONS",
    # Methods
    "BaseAttackMethod",
    "AttackOutput",
    "NoAttack",
    "SimpleParaphrase",
    "StealthRLAttack",
    "AdversarialParaphrasing",
    "AuthorMist",
    "HomoglyphAttack",
    "HomoglyphSweep",
    "sanitize_text",
    "get_method",
    "METHOD_REGISTRY",
    "GUIDANCE_VARIANTS",
    # Metrics
    "compute_auroc",
    "compute_tpr_at_fpr",
    "compute_asr",
    "compute_bootstrap_ci",
    "compute_detector_metrics",
    "compute_quality_metrics",
    "calibrate_thresholds",
    "DetectorMetrics",
    "QualityMetrics",
    "E5SimilarityScorer",
    "PerplexityScorer",
    # Plots
    "COLORBLIND_COLORS",
    "METHOD_NAMES",
    "COLORS",
    "create_heatmap",
    "create_tradeoff_plot",
    "create_budget_sweep_plot",
    "create_auroc_bar_chart",
    "create_auroc_radar_chart",
    "create_method_comparison_summary",
    "create_roc_curves",
    "create_roc_curves_logscale",
    "create_transferability_heatmap",
    "create_quality_likert_chart",
    "create_winrate_chart",
    "create_perplexity_comparison_table",
    "create_combined_results_table",
    "create_score_distribution_plot",
    "create_score_shift_plot",
    "create_human_ai_separation_plot",
    "create_main_results_table",
    "create_transfer_table",
    "create_quality_table",
    "generate_all_plots",
    "generate_all_tables",
    "QUALITY_RATING_PROMPT",
    "WIN_RATE_PROMPT",
    "get_quality_rating_messages",
    "get_win_rate_messages",
    "parse_quality_rating_response",
    "parse_win_rate_response",
    # Runner
    "EvalRunner",
    # Sanitization
    "sanitize",
    "compute_sanitization_diff",
    "run_sanitize_evaluation",
    "create_sanitize_report",
    "SanitizationResult",
    "ZERO_WIDTH_CHARS",
    "HOMOGLYPH_MAP",
    # Enhanced Runner
    "RunManager",
    "EnhancedEvalRunner",
]
