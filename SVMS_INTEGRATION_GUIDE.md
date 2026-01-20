# SVMS Integration Guide for GR00T N1.6 on RoboCasa

## ✅ Completed Components

### 1. **Core Sheaf Module** (`gr00t/model/modules/sheaf_streams.py`)
- ✅ `StreamHead`: 3 specialized processing heads
- ✅ `LowRankAdapter`: Sheaf restriction maps with low-rank factorization
- ✅ `SheafConsistency`: Loss computation + iterative correction
- ✅ `StreamRouter`: Adaptive token-level routing
- ✅ `SVMSWrapper`: Main wrapper integrating all components

### 2. **Auxiliary Labels** (`gr00t/data/robocasa_auxiliary_labels.py`)
- ✅ RoboCasa-specific keyword sets (objects, actions, states)
- ✅ Token classification functions (visual, temporal, state)
- ✅ Label generator from token IDs
- ✅ Coverage analysis utilities

### 3. **Config Updates** (`gr00t/configs/model/gr00t_n1d6.py`)
- ✅ Added 20+ SVMS configuration parameters
- ✅ Stream architecture settings (d_stream, d_overlap, adapter_rank)
- ✅ Sheaf loss scheduling (adaptive/linear/fixed modes)
- ✅ Auxiliary supervision parameters
- ✅ Router configuration (temperature schedule, dropout, balance)

---

## 🚧 Remaining Tasks

### 4. **Model Integration** (Next Step)
**File:** `gr00t/model/gr00t_n1d6/gr00t_n1d6.py`

**Injection Point:** Between backbone and action_head

**Current Flow:**
```
backbone(inputs) → backbone_outputs → action_head(backbone_outputs, action_inputs)
```

**SVMS Flow:**
```
backbone(inputs) → backbone_outputs → SVMS_wrapper(backbone_outputs) → action_head(refined_outputs, action_inputs)
```

**Required Changes:**

1. **Import SVMS modules** (top of file)
   ```python
   from gr00t.model.modules.sheaf_streams import SVMSWrapper
   ```

2. **Initialize SVMS in `Gr00tN1d6.__init__`** (after action_head initialization)
   ```python
   # Initialize SVMS wrapper if enabled
   if config.use_sheaf_streams:
       self.svms_wrapper = SVMSWrapper(
           d_vlm=config.backbone_embedding_dim,  # 2048
           d_stream=config.d_stream,
           d_overlap=config.d_overlap,
           adapter_rank=config.adapter_rank,
           dropout=config.attn_dropout,
           use_aux_losses=config.use_aux_losses,
       )
       print(f"✓ SVMS enabled with {config.n_streams} streams")
   else:
       self.svms_wrapper = None
   ```

3. **Modify `forward` method** (around line 510)
   ```python
   # Get backbone outputs
   backbone_outputs = self.backbone(backbone_inputs)

   # Apply SVMS if enabled
   if self.svms_wrapper is not None:
       svms_outputs = self.svms_wrapper(
           backbone_features=backbone_outputs["backbone_features"],
           attention_mask=backbone_inputs.get("attention_mask"),
           sheaf_unroll_steps=self.config.sheaf_unroll_steps,
           router_temperature=self._compute_router_temperature(step),
           aux_labels_A=inputs.get("aux_labels_A"),  # From data collator
           aux_labels_B=inputs.get("aux_labels_B"),
           aux_labels_C=inputs.get("aux_labels_C"),
       )

       # Replace backbone features with refined SVMS features
       backbone_outputs["backbone_features"] = svms_outputs["refined_features"]

       # Store SVMS outputs for loss computation
       backbone_outputs["svms_outputs"] = svms_outputs

   # Continue to action head
   outputs = self.action_head(backbone_outputs, action_inputs)
   ```

4. **Add router temperature scheduler**
   ```python
   def _compute_router_temperature(self, step: int) -> float:
       """Compute router temperature based on training step"""
       if step >= self.config.router_temp_decay_steps:
           return self.config.router_temp_final

       frac = step / self.config.router_temp_decay_steps
       temp = self.config.router_temp_init + frac * (
           self.config.router_temp_final - self.config.router_temp_init
       )
       return temp
   ```

### 5. **Trainer Modifications**
**File:** `gr00t/experiment/trainer.py`

**Required Changes:**

1. **Add auxiliary label generation in data collator**
   - Hook into existing data preprocessing
   - Call `create_auxiliary_labels_from_ids()` for each batch
   - Add aux labels to batch dict

2. **Extend loss computation**
   ```python
   # Existing diffusion loss
   loss = outputs.loss

   # Add SVMS losses if enabled
   if self.config.use_sheaf_streams and "svms_outputs" in outputs:
       svms = outputs.svms_outputs

       # Sheaf consistency loss (adaptive scheduling)
       lambda_sheaf = self._compute_sheaf_lambda(loss.item(), step)
       loss += lambda_sheaf * svms["sheaf_loss"]

       # Auxiliary loss (warmup schedule)
       lambda_aux = self._compute_aux_lambda(step)
       loss += lambda_aux * svms["aux_loss"]

       # Router regularization
       router_balance_loss = ((svms["router_weights"].mean(0) - 1/3)**2).sum()
       loss += self.config.router_balance_weight * router_balance_loss
   ```

3. **Add scheduling functions**
   ```python
   def _compute_sheaf_lambda(self, diffusion_loss, step):
       if self.config.sheaf_schedule_mode == "adaptive":
           if diffusion_loss > self.config.sheaf_delay_until_diffusion:
               return self.config.lambda_sheaf_min
           else:
               progress = (self.config.sheaf_delay_until_diffusion - diffusion_loss) / \
                         self.config.sheaf_delay_until_diffusion
               progress = max(0.0, min(1.0, progress))
               return self.config.lambda_sheaf_min + progress * (
                   self.config.lambda_sheaf_max - self.config.lambda_sheaf_min
               )
       # ... other modes
   ```

### 6. **Training Scripts**
**Files to Create:**

1. `scripts/train_svms_robocasa_phase1.sh` - Stream specialization
2. `scripts/train_svms_robocasa_phase2.sh` - Sheaf + partial unfreezing
3. `scripts/train_svms_robocasa_phase3.sh` - End-to-end fine-tuning

### 7. **Evaluation Scripts**
**Files to Create:**

1. `gr00t/eval/open_loop_eval_sheaf.py` - Open-loop comparison
2. `scripts/compare_baseline_svms.py` - Result visualization

---

## 📊 Memory Budget for RTX 32GB

### Baseline GR00T N1.6:
- VLM backbone: ~2.5GB
- 32-layer DiT: ~450MB
- Projectors: ~100MB
- **Total: ~3GB**

### SVMS Overhead:
- 3 StreamHeads: ~150MB
- 4 Adapters (low-rank): ~80MB
- Router: ~10MB
- Merge layer: ~30MB
- Auxiliary heads: ~5MB
- **Total overhead: ~275MB**

### Training Memory (Phase 2 - Critical):
- Model: ~3.3GB
- Activations (batch=12): ~8GB
- Gradients: ~3.3GB
- Optimizer states (AdamW): ~6.6GB
- **Total: ~21GB** ✅ Fits in 32GB with margin!

### Safety Margins:
- Phase 1: batch_size=16 (streams only) → ~18GB
- Phase 2: batch_size=12 (+ DiT bottom) → ~24GB
- Phase 3: batch_size=8 (full model) → ~28GB

All phases fit comfortably in 32GB with mixed precision (BF16).

---

## 🎯 Next Immediate Steps

1. **Modify `gr00t_n1d6.py`** to inject SVMS wrapper
2. **Modify trainer** to add SVMS losses
3. **Create Phase 1 training script**
4. **Run proof-of-concept training** (1k steps)
5. **Create open-loop evaluation**
6. **Compare results**

---

## 📝 Quick Reference: Training Commands

### Phase 1: Stream Specialization (5k steps, ~8 hours on RTX)
```bash
CUDA_VISIBLE_DEVICES=0 uv run python gr00t/experiment/launch_train.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path <ROBOCASA_PATH> \
    --embodiment-tag ROBOCASA_PANDA_OMRON \
    --use-sheaf-streams \
    --max-steps 5000 \
    --global-batch-size 16 \
    --learning-rate 1e-4 \
    --lambda-aux 0.5 \
    --lambda-sheaf 0.0 \
    --freeze-dit
```

### Open-Loop Evaluation
```bash
uv run python gr00t/eval/open_loop_eval_sheaf.py \
    --baseline-model nvidia/GR00T-N1.6-3B \
    --svms-model ./checkpoints/svms_phase3/final \
    --dataset-path <ROBOCASA_PATH> \
    --n-samples 1000
```

---

## 🔍 Monitoring During Training

### Key Metrics to Watch:

**Phase 1 (Stream Specialization):**
- ✅ Auxiliary accuracy > 70% for all streams (indicates specialization)
- ✅ Diffusion loss decreasing
- ✅ Router entropy remains high (uniform routing early on)

**Phase 2 (Sheaf Activation):**
- ✅ Sheaf residual decreasing (0.5 → 0.1)
- ✅ Sheaf lambda ramping up (0.01 → 0.1)
- ✅ Auxiliary accuracy stays > 65% (streams maintain specialization)
- ✅ Router starts specializing (entropy decreases, task-specific patterns)

**Phase 3 (End-to-End):**
- ✅ All losses stabilizing
- ✅ Open-loop MSE improving over baseline
- ✅ Sheaf residual < 0.15

---

## ⚠️ Potential Issues & Solutions

### Issue 1: Streams Collapse (all become identical)
**Symptoms:** Aux accuracy drops below 60%, sheaf residual → 0 too fast

**Solutions:**
- Lower `lambda_sheaf` (0.1 → 0.05)
- Increase `lambda_aux` (0.3 → 0.5)
- Add diversity loss

### Issue 2: OOM (Out of Memory)
**Symptoms:** CUDA OOM error during training

**Solutions:**
- Reduce batch size (16 → 12 → 8)
- Enable gradient checkpointing
- Reduce `d_stream` (768 → 512)

### Issue 3: Sheaf Hurts Performance
**Symptoms:** Diffusion loss increases when sheaf is activated

**Solutions:**
- Delay sheaf activation more (sheaf_delay_until_diffusion: 0.4 → 0.3)
- Lower max lambda (0.1 → 0.05)
- Use gentler correction (unroll_steps: 1 → 0)

---

## 📚 Code Organization

```
Isaac-GR00T/
├── gr00t/
│   ├── model/
│   │   ├── modules/
│   │   │   ├── sheaf_streams.py          ✅ CREATED
│   │   │   └── ... (existing modules)
│   │   └── gr00t_n1d6/
│   │       ├── gr00t_n1d6.py             🚧 TO MODIFY
│   │       └── ...
│   ├── configs/
│   │   └── model/
│   │       └── gr00t_n1d6.py             ✅ UPDATED
│   ├── data/
│   │   ├── robocasa_auxiliary_labels.py  ✅ CREATED
│   │   └── ...
│   ├── experiment/
│   │   ├── trainer.py                    🚧 TO MODIFY
│   │   └── ...
│   └── eval/
│       ├── open_loop_eval_sheaf.py       📝 TO CREATE
│       └── ...
├── scripts/
│   ├── train_svms_robocasa_phase1.sh     📝 TO CREATE
│   ├── train_svms_robocasa_phase2.sh     📝 TO CREATE
│   ├── train_svms_robocasa_phase3.sh     📝 TO CREATE
│   └── compare_baseline_svms.py          📝 TO CREATE
└── SVMS_INTEGRATION_GUIDE.md             ✅ THIS FILE
```

---

## 🎓 Key Architectural Decisions

1. **Injection Point:** Between backbone and DiT (minimal disruption to pretrained components)
2. **Stream Specializations:** Visual, Temporal, State (aligned with robotic manipulation)
3. **Phased Training:** Prevents catastrophic interference with pretrained weights
4. **Adaptive Sheaf Scheduling:** Only enforce consistency after basic task learning
5. **Low-Rank Adapters:** Parameter-efficient overlap projections
6. **Token-Level Routing:** More granular than sequence-level for mixed-modality tasks

---

**Status:** Ready for model integration step!
**Next File to Edit:** `gr00t/model/gr00t_n1d6/gr00t_n1d6.py`
