# TinyNet

TinyNet is a hardware-oriented multi-task 1D CNN for **atrial fibrillation (AF) detection** and **heart-rate (HR) estimation** from unified temporal windows. The project is organized around one consistent pipeline: preprocessing, balanced split construction, staged (decoupled) training, and post-training INT8 export.

## Model Architecture

![TinyNet Architecture](TinyNet/Tinynet_Arch_V9.jpg)

TinyNet uses a **shared stem + dual-branch head** design:

- Input is a fixed window with shape `(C=16, L=320)`.
- A shared stem (`Conv1d(7) + BN + ReLU`) extracts base features.
- The network then splits into two task branches:
  - **HR branch (regression via distribution logits)**
    - `ResBlock x3` -> `1x1 Conv` -> cascaded GAP -> Dropout -> FC
    - Outputs HR logits over BPM bins.
  - **AF branch (binary classification)**
    - **Spatial stream:** `ResBlock x3 + BN`
    - **Temporal stream:** pooled compression -> `ResBlock x2` -> data/gate conv pair -> sigmoid gating -> global temporal pooling
    - Fusion: spatial-temporal add -> SE reweighting -> AF head -> cascaded GAP -> Dropout -> FC(1)
- Quantization stubs are built into the model for INT8 deployment flow.

Design constraints are hardware-first: channel counts are enforced as multiples of 16, operator choices are restricted to supported kernel/stride patterns, and HR output dimension is capped for deployment limits.

## Project File Tree

```text
TinyNet/
├── config/
│   └── config.yaml                  # Central configuration for all stages
├── scripts/
│   ├── preprocess_*.py              # Dataset-specific preprocessing entrypoints
│   ├── train_decoupled.py           # 3-stage decoupled training entrypoint
│   ├── evaluate_fp32.py             # FP32 evaluation entrypoint
│   ├── evaluate_int8.py             # INT8 evaluation entrypoint
│   ├── export_TinyNet.py            # PTQ + INT8/hex/graph export pipeline
│   └── plot_tinynet_architecture.py # Architecture graph drawing utility
├── src/
│   ├── data/
│   │   ├── preprocessing.py         # Filtering, normalization, windowing, label building
│   │   └── dataloader.py            # Dataset class, augmentation, weighted sampler
│   ├── models/
│   │   └── tinynet.py               # TinyNet backbone/heads and hardware checks
│   ├── loss/
│   │   └── loss.py                  # Integrated AF + HR loss with uncertainty weights
│   ├── train/
│   │   ├── train.py                 # Stage loop, validation metrics, checkpointing
│   │   └── helper.py                # Seed/init/freeze/threshold/early-stop helpers
│   ├── evaluate/
│   │   ├── evaluate.py              # Unified evaluation and report generation
│   │   └── helper.py                # Post-processing, temporal decoder, plots
│   └── utils/
│       ├── config.py                # Typed config accessor and device resolver
│       ├── generate_split.py        # Subject-level balanced split generation
│       ├── quant_export_utils.py    # FX hooks and hardware hex export utilities
│       └── visualization.py         # Training visualization helper
├── split_optimized.json             # Train/val/test split + sampler weights
├── Tinynet_Arch_V9.jpg              # Architecture figure used in this README
├── checkpoints/                     # Saved model checkpoints
├── output/                          # INT8 models, hex dumps, exported graphs
└── plots/                           # Evaluation and diagnosis figures
```

## End-to-End Workflow

### 1) Preprocessing

- Raw synchronized signals are filtered and normalized (band-pass, Hampel, z-score).
- Signals are resampled to a unified target rate and packed into fixed 16-channel windows.
- Labels are generated per window:
  - AF: binary one-hot
  - HR: Gaussian-smoothed distribution over BPM bins
- The pipeline writes paired arrays: `*_X.npy` (features) and `*_y.npy` (labels).

### 2) Balanced Split + Sampling Setup

- `generate_split.py` builds subject-isolated train/val/test partitions.
- Split quality is optimized using class/bucket balance scoring.
- The split file also stores sampler weights for:
  - AF/HR task balance
  - AF positive/negative rebalance
  - HR low/mid/high bucket rebalance

### 3) Decoupled Training (Three Stages)

Training is intentionally staged to reduce task interference:

- **Stage 1 (joint warm-up):** train both AF and HR branches together.
- **Stage 2 (AF-focused):** freeze HR branch, prioritize AF optimization and AF threshold behavior.
- **Stage 3 (HR-focused):** freeze AF branch, optimize HR error metrics.

Across stages, training uses:

- task-masked integrated loss (AF BCE + HR distribution loss),
- learnable uncertainty-based task weighting,
- stage-specific weighted sampling,
- early stopping and checkpoint selection by stage objective.

### 4) Quantization and Export

- A trained FP32 checkpoint is converted through PTQ (QNNPACK backend).
- Calibration windows are selected using configurable strategy (default: stratified by source).
- Quantization-safe graph preparation includes selective module fusion and SE/BN handling for stable conversion.
- Export artifacts include:
  - quantized deployable model,
  - FX folder export,
  - per-layer activation/weight/bias hex dumps,
  - graph visualization for hardware inspection.

## Strategy Used Across the Whole Pipeline

- **Hardware-first modeling:** architecture and dimensions are constrained from the start to satisfy deployment rules.
- **Task decoupling for stability:** AF and HR are first co-trained, then optimized in dedicated stages to reduce gradient conflict.
- **Imbalance-aware learning:** split generation and weighted sampling are both class/bucket aware, not only random-shuffle based.
- **Deployment-consistent training path:** calibration data handling, quantization-aware module preparation, and export tooling are integrated as first-class pipeline stages, not post-hoc scripts.

