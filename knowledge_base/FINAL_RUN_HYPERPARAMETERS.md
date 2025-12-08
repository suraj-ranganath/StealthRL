# Final Run Hyperparameters - Optimized Configuration

**Date**: December 7, 2025  
**Model**: Qwen/Qwen3-4B-Instruct-2507  
**Dataset**: tinker_large (4,625 train, 1,157 test)  
**Sources**: Thinking Machines LoRA research, Tinker Cookbook, GRPO-RL-Training SKILLs

---

## ✅ CRITICAL CHANGES FROM DEFAULT

### 1. **LoRA Configuration** (MOST IMPORTANT)

```yaml
lora:
  rank: 32              # ↑ CHANGED from 16
  alpha: 32             # ✓ FIXED to 32 (was null)
  target_modules: null  # ✓ CONFIRMED (all layers including MLP)
```

**Justification**:
- **Rank=32**: Thinking Machines research shows rank=32 is **optimal for RL**
  - Even rank=1 works for RL (information-theoretic argument: O(1) bits per episode)
  - Rank=32 matches full fine-tuning performance
  - Higher ranks don't improve performance but slow training
  
- **Alpha=32**: Standard scaling factor
  - DO NOT scale alpha with rank (keeps optimal LR independent of rank)
  - α/r prefactor makes learning curves identical across ranks early in training
  
- **All layers**: Apply LoRA to **ALL weight matrices** (CRITICAL)
  - MLP layers contain most parameters (70%+ in Qwen models)
  - Attention-only LoRA **significantly underperforms** even with matched param count
  - Research: "MLP-only > MLP+Attention ≈ Full > Attention-only"

---

### 2. **Learning Rate** (CRITICAL)

```yaml
training:
  learning_rate: 2.8e-4  # ↑ CHANGED from 1e-5
```

**Justification**:
- **10x FullFT rule**: LoRA requires **10x higher LR** than full fine-tuning
  - Empirical result from Thinking Machines across 14 models
  - Formula: `LR = 5e-5 * 10 * (2000/hidden_size)^0.781`
  - For Llama-3.1-8B (hidden=4096): LR ≈ 2.8e-4
  - For Qwen3-4B (hidden=3584): LR ≈ 3.2e-4, but 2.8e-4 is conservative and safe
  
- **Why 10x?**: LoRA initialization (B=0, A~Uniform) creates implicit LR schedule
  - Early training: B ≈ 0, so updates to A have minimal effect
  - Effective LR ramps up as B grows → need higher nominal LR
  
- **Rank independence**: Optimal LR is nearly independent of rank (due to 1/r scaling)
  - Same LR works for rank=1 to rank=512

---

### 3. **Batch Size** (CRITICAL)

```yaml
training:
  batch_size: 4     # ↓ CHANGED from 8
  group_size: 4     # ↑ CHANGED from 4
```

**Justification**:
- **Small batch_size=4**: LoRA is **less tolerant of large batch sizes** than full FT
  - Research shows LoRA pays larger penalty at batch_size > 32
  - Product-of-matrices (BA) parametrization has worse dynamics than W at large batches
  - Both LoRA and FullFT achieve best loss at **small batches** (4-16)
  
- **group_size=8**: GRPO sweet spot
  - GRPO research: 4-16 rollouts per prompt optimal
  - 8 gives good variance reduction without excessive compute
  - Effective batch = batch_size × group_size = 4 × 8 = 32 generations per step

---

### 4. **Temperature Schedule** (IMPORTANT)

```yaml
sampling:
  temperature: 1.0
  temperature_schedule: "constant"  # CHANGED from "decay"
  top_p: 0.95                       # ↑ CHANGED from 0.9
```

**Justification**:
- **Constant temperature**: GRPO research recommends **no decay** during RL training
  - Decay reduces exploration → worse policy diversity
  - Temperature=1.0 maintains proper entropy for reward estimation
  
- **top_p=0.95**: Slightly higher for more diversity
  - Helps explore paraphrase space
  - Prevents mode collapse

---

### 5. **KL Penalty** (IMPORTANT)

```yaml
kl:
  penalty_coef: 0.01  # ↑ CHANGED from 0.001
```

**Justification**:
- **0.01 vs 0.001**: Tinker research shows **0.001 is too weak**
  - Models drift too far from reference policy
  - Loss of fluency and coherence
  - Target KL should stay < 0.01 for stability
  
- **Fixed penalty**: Better than adaptive for GRPO
  - Adaptive KL complicates reward signal
  - Fixed β gives more stable training

---

### 6. **Advantage Clipping**

```yaml
grpo:
  advantage_clip: 10.0  # ↑ CHANGED from 5.0
```

**Justification**:
- GRPO research: clip=5.0 is **too conservative**
  - Limits learning from high-reward examples
  - 10.0 gives better gradient signal while still preventing outliers

---

## 📊 Expected Training Dynamics

### Metrics to Monitor

1. **Reward Metrics** (MOST IMPORTANT):
   - `reward_mean`: Should increase steadily (target: 0.5 → 0.7+)
   - `reward_std`: Should stay > 0.1 (mode collapse if < 0.05)
   - `detector_reward`: Should increase (0.3 → 0.6+)
   - `semantic_reward`: Should stay high (> 0.8)
   
2. **KL Divergence** (CRITICAL):
   - `kl_sample_train`: Should stay < 0.01 (fluency preserved)
   - `kl_post_train`: Post-update KL (should be < 0.02)
   - If KL > 0.01: model drifting too fast → increase kl.penalty_coef
   
3. **Loss** (COUNTER-INTUITIVE):
   - **Loss WILL GO UP** - This is NORMAL for RL!
   - Loss measures KL(π_new || π_ref), which increases as policy improves
   - DO NOT judge training by loss - use reward metrics!
   
4. **Format Rate**:
   - Should stay > 0.95 (valid outputs)
   - If drops: adjust reward weights or add format reward

### Expected Timeline

With 4,625 training samples, batch_size=4, group_size=8, 3 epochs:

```
Total steps ≈ (4625 / 4) * 3 = ~3,468 steps
Eval every 100 steps → 34 evals
Save every 500 steps → 6 checkpoints
Total time estimate: 2-3 hours (Tinker hosted)
```

### Convergence Signs

- **Good training**:
  - Reward increasing steadily
  - KL < 0.01
  - Reward std > 0.1
  - Semantic similarity > 0.88
  
- **Problems**:
  - Reward std < 0.05 → Mode collapse (reduce LR or increase exploration)
  - KL > 0.01 → Too fast drift (increase kl.penalty_coef to 0.02)
  - Reward plateaus early → LR too low or reward functions poorly designed

---

## 🎯 Hyperparameter Summary Table

| Parameter | Old Value | New Value | Reason |
|-----------|-----------|-----------|--------|
| **LoRA rank** | 16 | **32** | Optimal for RL per research |
| **LoRA alpha** | null (=rank) | **32** | Standard scaling (LR independence) |
| **Learning rate** | 1e-5 | **2.8e-4** | 10x FullFT rule for LoRA |
| **Batch size** | 8 | **4** | LoRA penalty at large batches |
| **Group size** | 4 | **8** | GRPO optimal variance reduction |
| **Temperature schedule** | decay | **constant** | Maintain RL exploration |
| **top_p** | 0.9 | **0.95** | More diversity |
| **KL penalty** | 0.001 | **0.01** | Prevent excessive drift |
| **Advantage clip** | 5.0 | **10.0** | Better gradient signal |

---

## 🔬 Research Citations

### Thinking Machines - LoRA Without Regret
- **URL**: https://thinkingmachines.ai/blog/lora/
- **Key findings**:
  - LoRA needs 10x higher LR than FullFT (empirical across 14 models)
  - Rank=32 optimal for RL (even rank=1 works due to O(1) bits per episode)
  - Apply LoRA to ALL layers (MLP critical, attention-only fails)
  - Small batches better for LoRA (< 32)
  - Optimal LR independent of rank (due to α/r scaling)

### Tinker Cookbook - AGENTS.md
- **URL**: https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/AGENTS.md
- **Key patterns**:
  - Pipeline batches (submit forward_backward + optim_step together)
  - Use importance_sampling loss for policy gradient RL
  - Monitor KL divergence (should stay < 0.01)
  - LR formula: `hyperparam_utils.get_lr(model_name)`

### GRPO-RL-Training SKILLs
- **URL**: https://github.com/zechenzhangAGI/AI-research-SKILLs/tree/main/06-post-training/grpo-rl-training
- **Key insights**:
  - Loss goes UP during RL training (normal!)
  - Use 3-5 reward functions (single rewards fail)
  - Monitor reward_std > 0.1 (mode collapse detection)
  - num_generations=4-8 optimal (group_size)
  - Constant temperature for RL (no decay)

---

## ⚠️ CRITICAL REMINDERS

1. **Loss goes UP** - This is normal for RL! Judge by rewards, not loss.

2. **Monitor KL < 0.01** - If KL exceeds 0.01, increase `kl.penalty_coef` to 0.02

3. **Reward std > 0.1** - If < 0.05, mode collapse occurring (reduce LR or increase temp)

4. **Test rewards first** - Before full run, validate each reward function works correctly

5. **Small batches better** - batch_size=4 optimal for LoRA (research-backed)

6. **LR is 10x FullFT** - 2.8e-4 is correct for LoRA on Llama/Qwen models

7. **Apply LoRA to ALL layers** - MLP layers critical (attention-only fails badly)

---

## 📈 Post-Training Validation

After training completes, verify:

1. **Model quality**:
   - ASR > 60% on ensemble detectors
   - Semantic similarity > 0.88
   - ESL FPR gap < 0.07
   
2. **Training stability**:
   - Final KL < 0.01
   - No sudden reward collapse
   - Reward std stayed > 0.1 throughout
   
3. **Checkpoints saved**:
   - checkpoint-1000.pt
   - checkpoint-1500.pt
   - checkpoint-2000.pt
   - checkpoint-2500.pt
   - checkpoint-3000.pt
   - checkpoint-final.pt

---

## 🚀 Ready to Launch

All hyperparameters optimized based on:
✅ Thinking Machines LoRA research (2025)  
✅ Tinker Cookbook best practices  
✅ GRPO-RL-Training SKILLs  
✅ 4,625 training samples (tinker_large dataset)  

**Estimated training time**: 2-3 hours on Tinker hosted  
**Expected outcome**: ASR 60-70%, Semantic >0.88, ESL gap <0.07

Good luck with your final run! 🎯
