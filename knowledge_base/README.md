# StealthRL Knowledge Base

This directory contains comprehensive documentation for the StealthRL project. All documentation files have been organized here for easy reference and maintenance.

---

## 📚 Documentation Index

### Getting Started

1. **[QUICKSTART.md](QUICKSTART.md)** - Fast-track guide to run research experiments
   - Quick automated pipeline
   - Step-by-step commands
   - Expected results and validation

2. **[SETUP_AND_RUN.md](SETUP_AND_RUN.md)** - Complete setup guide with API keys
   - Tinker API key setup
   - Environment configuration
   - Full research pipeline execution
   - Troubleshooting and debugging

3. **[QUICK_START_RUNS.md](QUICK_START_RUNS.md)** - Quick start runs and experiments
   - Fast experiment configurations
   - Testing and validation runs

### Platform Integration

4. **[TINKER_README.md](TINKER_README.md)** - Tinker platform integration guide
   - Tinker API details
   - GRPO training with Qwen3-4B
   - Remote compute setup
   - DSC 291 deployment instructions

### Core Implementation

5. **[CHECKPOINT_GUIDE.md](CHECKPOINT_GUIDE.md)** - Checkpoint management guide
   - Saving and loading checkpoints
   - Model export and inference
   - Checkpoint formats and usage

6. **[CHECKPOINT_IMPLEMENTATION.md](CHECKPOINT_IMPLEMENTATION.md)** - Technical checkpoint implementation
   - Implementation details
   - Tinker checkpoint API
   - Code examples and patterns

7. **[REWARD_REFINEMENT.md](REWARD_REFINEMENT.md)** - Reward function design and refinement
   - Multi-objective reward formula
   - Normalization strategies
   - Session 4 enhancements
   - KL regularization

8. **[DETECTOR_SETUP.md](DETECTOR_SETUP.md)** - AI detector configuration
   - Detector ensemble setup
   - Fast-DetectGPT, Ghostbuster, Binoculars
   - Caching and optimization

### Evaluation and Fairness

9. **[ESL_FAIRNESS_GUIDE.md](ESL_FAIRNESS_GUIDE.md)** - ESL fairness evaluation
   - ESL vs native corpus
   - Fairness metrics (FPR gaps)
   - BERTScore integration
   - Evaluation pipeline

10. **[IMPLEMENTATION_VERIFICATION.md](IMPLEMENTATION_VERIFICATION.md)** - Implementation verification checklist
    - Code completeness checks
    - Testing procedures
    - Validation steps

### Research Planning

11. **[RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md)** - Research roadmap and timeline
    - Experimental design
    - Research questions
    - Expected outcomes
    - Future directions

12. **[NEXT_STEPS.md](NEXT_STEPS.md)** - Next steps and TODOs
    - Immediate actions
    - Pending experiments
    - Future enhancements

### Operations

13. **[RUN_MANAGEMENT.md](RUN_MANAGEMENT.md)** - Training run management
    - Experiment tracking
    - Output organization
    - Log management
    - Result analysis

---

## 🎯 Quick Navigation

### I want to...

**Run my first experiment:**
→ Start with [SETUP_AND_RUN.md](SETUP_AND_RUN.md) to set up API keys, then [QUICKSTART.md](QUICKSTART.md) for automated pipeline

**Understand the platform:**
→ Read [TINKER_README.md](TINKER_README.md) for Tinker integration details

**Work with checkpoints:**
→ See [CHECKPOINT_GUIDE.md](CHECKPOINT_GUIDE.md) for user guide, [CHECKPOINT_IMPLEMENTATION.md](CHECKPOINT_IMPLEMENTATION.md) for technical details

**Evaluate fairness:**
→ Follow [ESL_FAIRNESS_GUIDE.md](ESL_FAIRNESS_GUIDE.md) for ESL evaluation setup

**Understand reward design:**
→ Read [REWARD_REFINEMENT.md](REWARD_REFINEMENT.md) for reward function details

**Set up detectors:**
→ Check [DETECTOR_SETUP.md](DETECTOR_SETUP.md) for detector configuration

**Plan experiments:**
→ Review [RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md) and [NEXT_STEPS.md](NEXT_STEPS.md)

---

## 📝 Main Project Files (in root directory)

- **[README.md](../README.md)** - Project overview and main documentation
- **[REPORT.md](../REPORT.md)** - Comprehensive project report with all experiments and findings
- **[interaction_records.md](../interaction_records.md)** - Detailed development log and session history

---

## 🔄 Documentation Maintenance

All documentation in this directory should be:
- **Up to date** with latest implementation
- **Cross-referenced** where appropriate
- **Example-driven** with code snippets and commands
- **Practical** with clear actionable steps

When adding new documentation:
1. Create the file in this directory
2. Add entry to this index (with description)
3. Update cross-references in other docs
4. Add to "Quick Navigation" if relevant

---

**Last Updated:** Session 9 - Documentation reorganization  
**Total Documentation Files:** 13 guides + this index
