# SVMS-GR00T Quick Start Guide

## 🚀 From Zero to Training in 5 Steps

### Step 1: Get RoboCasa Data (1-2 hours)

```bash
cd Isaac-GR00T
bash gr00t/eval/sim/robocasa/setup_RoboCasa.sh
# Follow prompts to download or collect demonstrations
```

### Step 2: Prepare Dataset (10 min - 1 hour)

```bash
python scripts/prepare_robocasa_for_groot.py \
    --input /path/to/raw/robocasa \
    --output ./data/robocasa_groot \
    --validate \
    --visualize-samples 5
```

### Step 3: Update Config (5 min)

```bash
# Copy normalization stats from ./data/robocasa_groot/meta.json
# Paste into gr00t/configs/data/robocasa_modality_config.py
nano gr00t/configs/data/robocasa_modality_config.py
```

### Step 4: Start Phase 1 Training (8 hours)

```bash
# Edit script to set dataset path
nano scripts/train_svms_robocasa_phase1_poc.sh
# Change: DATASET_PATH="./data/robocasa_groot"

# Run training
bash scripts/train_svms_robocasa_phase1_poc.sh
```

### Step 5: Monitor Progress

```bash
# Check W&B dashboard
# Look for:
#   - aux_acc_A > 70%
#   - aux_acc_B > 70%
#   - aux_acc_C > 65%
```

---

## 📊 Training Timeline

```
Day 1:  Phase 1 (8h)  → Stream specialization
Day 2:  Phase 2 (16h) → Sheaf activation
Day 3:  Phase 3 (10h) → End-to-end fine-tuning
Total: ~34 hours GPU time
```

---

## 🔍 Key Files Reference

| File | Purpose | Location |
|------|---------|----------|
| Dataset Processor | Convert coordinates | `gr00t/data/robocasa_dataset_processor.py` |
| Prep Script | CLI for processing | `scripts/prepare_robocasa_for_groot.py` |
| Config | Modality settings | `gr00t/configs/data/robocasa_modality_config.py` |
| Phase 1 Script | First training | `scripts/train_svms_robocasa_phase1_poc.sh` |
| Phase 2 Script | Second training | `scripts/train_svms_robocasa_phase2.sh` |
| Phase 3 Script | Final training | `scripts/train_svms_robocasa_phase3.sh` |
| Eval Script | Testing | `gr00t/eval/open_loop_eval_sheaf.py` |

---

## ⚙️ Critical Parameters

### Dataset Preparation:
- `--action-horizon 16` (GR00T default)
- `--use-relative-actions` (REQUIRED!)

### Phase 1 Training:
- `LAMBDA_AUX=0.5` (strong supervision)
- `LAMBDA_SHEAF=0.0` (OFF - let streams diverge)

### Phase 2 Training:
- `LAMBDA_AUX=0.3` (continue supervision)
- `LAMBDA_SHEAF_MAX=0.1` (activate sheaf)
- `UNFREEZE_DIT_BOTTOM_LAYERS=8`

### Phase 3 Training:
- `LAMBDA_AUX=0.2` (reduce supervision)
- `LAMBDA_SHEAF_MAX=0.1` (constant)
- Full model unfrozen

---

## 🎯 Success Criteria

### After Phase 1:
✅ Stream A (Visual) accuracy > 70%
✅ Stream B (Temporal) accuracy > 70%
✅ Stream C (State) accuracy > 65%
✅ Streams are specialized (not collapsed)

### After Phase 2:
✅ Sheaf loss < 0.1
✅ Router weights balanced (~33% each)
✅ Action accuracy improved over Phase 1

### After Phase 3:
✅ End-to-end loss decreased
✅ Open-loop accuracy > baseline GR00T
✅ Ready for closed-loop deployment

---

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| OOM in Phase 1 | Reduce `GLOBAL_BATCH_SIZE` to 12 |
| OOM in Phase 2 | Reduce to 12, increase `GRADIENT_ACCUM_STEPS` to 6 |
| OOM in Phase 3 | Reduce to 8, add `--gradient-checkpointing` |
| Sheaf loss not decreasing | Check that Phase 1 specialized correctly |
| All aux accuracies < 50% | Check auxiliary label generation |
| NaN in training | Reduce learning rate by 2x |

---

## 📞 Need Help?

1. Check `IMPLEMENTATION_COMPLETE.md` for full details
2. Check `SVMS_INTEGRATION_GUIDE.md` for architecture
3. Check `ROBOCASA_SETUP_COMPLETE.md` for dataset processing
4. Check training script comments for phase-specific details

---

## ✅ Pre-Flight Checklist

Before starting Phase 1:
- [ ] RoboCasa dataset downloaded/collected
- [ ] Dataset converted with `prepare_robocasa_for_groot.py`
- [ ] Normalization stats updated in `robocasa_modality_config.py`
- [ ] `DATASET_PATH` set in Phase 1 script
- [ ] W&B configured (or disabled with `--no-use-wandb`)
- [ ] GPU available with 32GB VRAM

---

**That's it! You're ready to train SVMS-GR00T!** 🎉
