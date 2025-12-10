# LaTeX Report Figures Guide

## Included Figures in report.tex

Your LaTeX report (`report.tex`) now includes four key figures from your training results:

### Figure 1: Training Curves (training_curves.png)
- **Location**: Section 5 (Results)
- **File**: `outputs/tinker_ultrafast/run_20251207_212110/visualizations/training_curves.png`
- **Shows**: 6-panel visualization of training progression
  - Total reward (train/test)
  - Detector evasion progress
  - Semantic similarity preservation
  - Perplexity naturalness
  - KL divergence with target threshold
  - Parse success rate

### Figure 2: Reward Decomposition (reward_decomposition.png)
- **Location**: Section 5 (Results)
- **File**: `outputs/tinker_ultrafast/run_20251207_212110/visualizations/reward_decomposition.png`
- **Shows**: 4-panel reward component analysis
  - Stacked positive reward contributions
  - Individual component trajectories
  - Detector probability distribution
  - Metric correlation heatmap

### Figure 3: Pareto Frontiers (pareto_frontiers.png)
- **Location**: Section 5 (Results)
- **File**: `outputs/tinker_ultrafast/run_20251207_212110/visualizations/pareto_frontiers.png`
- **Shows**: Multi-objective trade-off analysis
  - 2D: Stealth vs Quality (9 Pareto-optimal checkpoints)
  - 3D: Stealth × Quality × Naturalness (26 Pareto-optimal checkpoints)

### Figure 4: Stability Metrics (stability_metrics.png)
- **Location**: Section 4 (Method)
- **File**: `outputs/tinker_ultrafast/run_20251207_212110/visualizations/stability_metrics.png`
- **Shows**: Training stability and convergence
  - Policy entropy (exploration level)
  - Learning rate schedule
  - Generation length statistics
  - Iteration timing

## Compiling the Report

To compile the PDF:

```bash
cd /Users/suraj/Desktop/StealthRL
pdflatex report.tex
pdflatex report.tex  # Run twice to resolve references
```

The output will be `report.pdf` (14 pages).

## File Structure Requirements

All figure files must be in their original locations:
```
StealthRL/
├── report.tex
└── outputs/
    └── tinker_ultrafast/
        └── run_20251207_212110/
            └── visualizations/
                ├── training_curves.png
                ├── reward_decomposition.png
                ├── pareto_frontiers.png
                └── stability_metrics.png
```

## Notes

- All images are 300 DPI PNG files suitable for publication
- Figures are sized to `\textwidth` for optimal readability
- Each figure includes detailed captions explaining the visualization
- The report compiled successfully to a 14-page PDF
- Total file size: ~3.2 MB

## Submission Checklist

✅ LaTeX source file: `report.tex`
✅ All 4 figure files in correct locations
✅ PDF compiles without errors
✅ All references and citations formatted correctly
✅ Team contributions section included
✅ Abstract summarizes key results (22% improvement, 98.6% similarity, 9 Pareto checkpoints)

Your report is ready for submission!
