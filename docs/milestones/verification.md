# M1–M4 verification and handoff

**Verified:** October 21, 2026, Asia/Kolkata. Persisted runtime timestamps use UTC.

## Assessment evidence mapping

| Requirement | Delivered evidence |
|---|---|
| M1: identify/define the problem and analyze the domain | `m1-problem-analysis.md`: problem, stakeholders, scope, importance, data constraints, research questions |
| M2: existing systems and feasibility | `m2-planning-justification.md`: cited comparison of GAN/VQ/DDPM/latent/medical diffusion, technical/financial/time feasibility, risks |
| M2: objectives and methodology | Explicit objectives, architecture diagram, data/model contracts, ten-stage workflow |
| M2: relevance of techniques | Technique justification table, quantization/noising/objective definitions, declared baseline differences |
| M3: implemented design, data and baseline | Real MRNet acquisition, preprocessing and split metadata; trained VQ model; `m3-design-implementation.md` traceability |
| M4: working modules with trained model | Saved VQ/diffusion checkpoints, real validation/generation evidence, dashboard live reconstruction and fresh-seed generation |

This maps deliverables to the supplied rubric. Assessment marks and acceptance
remain the institution's decision. The observed generation limitations below
must accompany the trained-model demonstration.

## Data and trained models

- Public Kaggle source: `cjinny/mrnet-v1`, version 1; original source attribution
  and unspecified mirror-license metadata are recorded.
- **96 training / 24 validation exams**, official split assignment retained.
- **360 real acquired series**, all three planes, **694.3 MiB** of source arrays.
- Model input: sagittal `[1,128,128,32]`, min-max [-1,1].
- CUDA 12.8, PyTorch **2.11.0+cu128**, RTX 5070 Ti Laptop GPU.
- VQ-VAE: **40 epochs, 1,920 optimizer updates**; selected epoch **38**.
- Conditional latent DDPM with bottleneck attention: **300 epochs, 14,400 updates**;
  selected epoch **283**.
- Initial unbounded sampling failed through extreme latent amplitudes. Its
  outputs and checkpoint snapshot were retained under `sampling-ablation/`.
  Current sampling bounds predicted-clean latents to the trained codebook's
  per-channel envelope before the DDPM posterior update.
- Eight saved generated volumes; same-device repeated seeded inference produced
  byte-identical output. Different-seed live inference produced a distinct latent.

## Actual validation results

| Metric | Trained VQ | Half-resolution interpolation reference |
|---|---:|---:|
| MSE ↓ | 0.01537 | 0.02975 |
| MAE ↓ | 0.07493 | 0.08920 |
| PSNR ↑ | 24.29968 dB | 21.36067 dB |
| SSIM ↑ | 0.76587 | 0.77707 |

The learned model improves intensity errors and PSNR but has slightly lower
SSIM than this interpolation reference. The reference has a different information
budget and is not presented as an equivalent learned-compression competitor.
Validation was used for model selection; these are not independent test results.

Generated pairwise adaptive 3D MS-SSIM was **0.22963**, versus **0.30247** for the
eight selected real training references (window 3, four scales). Lower similarity
does not imply better images. Visual inspection shows blurred texture and
unreliable large-scale knee structure: realistic anatomical generation has not
been established. No downstream augmentation or clinical result is claimed.

## Executed verification

### Automated suite

Equivalent portable command (the recorded run used isolated temporary/cache directories):

```powershell
python -m pytest -q
```

Result: **49 passed, 1 skipped in 31.53 seconds**. The one local trained-model
acceptance check was skipped while its training artifacts were being finalized.

### Real-checkpoint live acceptance

Equivalent portable command:

```powershell
python -m pytest tests/test_m4_live.py -q -rs
```

Result: **1 passed in 7.24 seconds** after training completed. This drives the
actual Model Experiments page, executes the selected VQ model on a validation
input, changes the sampling seed to 2037, executes the diffusion/VQ models,
and validates the newly written volume and model lineage. Thus **all 50 checks
were executed successfully across these two runs**.

Other checks: `pip check` reported no broken requirements; a CUDA 3D convolution
forward/backward check passed; the live dashboard health endpoint returned
**HTTP 200, `ok`** on `http://127.0.0.1:8501/_stcore/health`.

## Files to present

- [Problem analysis](m1-problem-analysis.md)
- [Planning and justification](m2-planning-justification.md)
- [Design-to-implementation mapping](m3-design-implementation.md)
- [Exact M4 demonstration procedure](m4-demo.md)
- [Presentation slide content](presentation.md)
- [Published aggregate baseline results](../results/m4-baseline.md)
- Complete computed evidence is generated locally at `runs/M4-MRNET-20261021/evidence.md`.
- `runs/M4-MRNET-20261021/evidence-assets/`: actual training curves, three
  held-out reconstruction comparisons, and generated-volume montage.

Launch with `.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py`.
The dashboard binds to loopback on port 8501. A fresh clone requires explicitly
downloading data and reproducing training before the real-checkpoint demo is available.
