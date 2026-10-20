# M3 — Design and Implementation

**Assessment window:** October 12–16, 2026 · **Weight:** 3 marks

**Evidence pack prepared:** October 21, 2026

## 1. Design-to-code traceability

| Design component | Implementation | Observable evidence |
|---|---|---|
| Explicit data acquisition | `scripts/download_mrnet_subset.py` | Source manifest with version, selection seed, real labels, source sizes/hashes |
| Knee data adapter | `src/data/mrnet.py` | Original acquired planes, strict labels, official train/valid mapping |
| Generic 3D NIfTI layer | `src/data/loader.py`, `preprocessing.py`, `pipeline.py` | Unit-aware geometry, canonical orientation, crop/pad/resize, isolated exports |
| Reusable model inputs | `src/data/dataset.py`, `src/training/common.py` | `[B,1,row,column,slice]` knee batches; content verification and split checks |
| Baseline latent learning | `src/models/vqvae.py`, `train_vq()` | Real training updates, validation curve, selected and last checkpoints |
| 3D latent generator | `src/models/diffusion.py`, `train_diffusion()` | Noise schedule, conditional U-Net, optional bottleneck attention, loss curve |
| Sampling and decoding | `generate()` | Seeded noise, denormalized latent, learned-codebook projection, generated arrays |
| Evaluation | `src/metrics.py`, `src/training/evaluation.py` | Paired reconstruction metrics and unpaired generation diagnostics |
| Research interface | `app/streamlit_app.py`, `research_workspace.py` | Acquired-series browsing, reconstructions, generated samples, live inference |
| Submission evidence | `src/training/evidence.py` | Machine-generated tables, plots, fixed-slice comparisons, model hashes |

## 2. Data handling design

### Two supported input contracts

**NIfTI:** `[R,A,S]` after canonicalization, with affine/spatial units tracked
through preprocessing. Exports add a channel axis. Subject directories or an
explicit manifest determine leakage-safe seeded splits.

**MRNet:** released arrays are `[slice,row,column]`. The selected series becomes
`[1,row,column,slice]` after resizing and normalization. No physical spacing,
patient-space affine, or registration between acquired planes is invented.
Official train/valid assignments are retained; patient IDs are unavailable.

### Reduced training cohort

The configured downloader requests 96 training and 24 validation exams, with all
three acquired planes. Proportional joint-label strata and seeded selection
operate separately inside the official splits. This is a reduced academic
cohort, not the full MRNet benchmark. The actual downloaded counts, class
support, version, byte total, and source hashes are in:

`data/mrnet/MRNet-v1.0/source_manifest.json`

The sagittal series supplies the trained baseline. Preprocessing validates
finite 3D arrays, transposes, resizes to 128×128×32, normalizes to [-1,1], and
exports channel-first float32 arrays and a metadata CSV.

## 3. Baseline architecture

```text
[B,1,128,128,32]
  → stride-2 3D convolution + residual block
  → stride-2 convolution + residual block
  → 8-channel projection
  → nearest-neighbor vector codebook, 128 embeddings
[B,8,32,32,8]
  → 3D residual decoder + transposed convolutions
  → tanh
[B,1,128,128,32]
```

The two stride-2 convolutions give total ×4 per spatial axis.

The M4 trained baseline selects VQ-VAE (`gan_weight: 0`) for stable paired
reconstruction. VQ-GAN training is implemented as an optional extension. Training
features initialize the codebook, and validation L1 selects the saved best model.
The last checkpoint retains optimizer and random-generator state for resume.

## 4. Baseline comparison and verification

The learned VQ reconstruction is compared with an explicit **half-resolution
linear interpolation baseline**. Both operate on the same validation tensors
and intensity range. The interpolation reference is not an equivalent learned
compression system and may achieve higher similarity; results are reported
without hiding that outcome.

Automated checks cover malformed data, spatial operations, reproducible splits,
export equivalence, corrupted tensors, model gradients, checkpoint resume,
conditional sampling, attention gradients, and generation-evaluation lineage.
Tests use small numerical fixtures; separate integration checks use real MRI.

## 5. How to demonstrate M3

1. Launch the dashboard using the project's `.venv`.
2. Show **MRNet Dataset**: recorded source, label counts, and each actual acquired
   plane for one examination.
3. Show the processed `metadata.csv`, tensor dimensions, and official split counts.
4. Open **Model Experiments** and show actual VQ training/validation curves.
5. Compare a held-out input and its saved reconstruction; then run live reconstruction.
6. Open the generated `evidence.md` for measured metrics and fixed-slice images.

Use `docs/milestones/m4-demo.md` for the exact commands and complete demo sequence.
