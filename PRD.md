# PRD — 3D MRI Generative AI
## Milestone 1: Dataset, Preprocessing & 3D MRI Visualization Platform

**Project Type:** Final-Year AI/ML Research Project<br>
**Domain:** Healthcare / Medical Imaging / Generative AI<br>
**Primary Framework:** Python + PyTorch<br>
**Frontend:** Streamlit<br>
**Backend/ML:** Python<br>
**Target:** Research-ready foundation for 3D MRI diffusion modeling

---

# 1. Project Overview

We are building a research platform for **3D MRI image generation using Generative AI**.

The long-term objective is to train a latent diffusion model capable of generating realistic 3D MRI volumes. The architecture will eventually follow the general approach described in the Medical Diffusion research paper:

```text
3D MRI
   ↓
VQ-GAN / VQ-VAE Encoder
   ↓
Latent Representation
   ↓
3D Diffusion Model
   ↓
Generated Latent Representation
   ↓
VQ-GAN / VQ-VAE Decoder
   ↓
Generated 3D MRI
```

However, **Milestone 1 must NOT implement the full diffusion model**.

Milestone 1 focuses exclusively on building a reliable medical-imaging data pipeline that can:

- acquire/load MRI data
- validate MRI volumes
- preprocess MRI volumes
- normalize MRI intensity
- resize/crop MRI volumes
- split data correctly
- visualize 3D MRI volumes
- calculate dataset statistics
- provide baseline reconstruction/quality measurements
- prepare standardized data for future VQ-GAN and diffusion training

The output of Milestone 1 should be a clean, reproducible research codebase.

---

# 2. Problem Statement

Medical MRI scans are naturally 3D volumes consisting of multiple 2D slices.

A typical MRI volume can be represented as:

```text
Height × Width × Depth
```

For example:

```text
256 × 256 × 32
```

A major challenge in 3D medical-image generation is computational cost and maintaining anatomical consistency across slices.

Before implementing a generative model, we need a robust pipeline that converts raw MRI data into consistent 3D tensors suitable for deep-learning models.

The system must therefore establish a standardized MRI preprocessing and visualization pipeline.

---

# 3. Milestone 1 Objective

Build a research-grade application that allows a researcher to:

1. Load MRI data.
2. Inspect MRI metadata.
3. Validate MRI volumes.
4. Preprocess MRI volumes.
5. Normalize intensities.
6. Resize/crop volumes to a consistent shape.
7. Visualize individual MRI slices.
8. Visualize the complete volume.
9. Inspect intensity distributions.
10. Compare original and processed MRI data.
11. Export processed data.
12. Generate a dataset-quality report.

The resulting processed dataset will become the input for **Milestone 2: VQ-VAE/VQ-GAN latent representation learning**.

---

# 4. Target User

The primary user is:

**AI/ML researcher or final-year student working on medical image generation.**

The application scope is local MRI research and experimentation.

It is a **research and data-processing tool only**.

---

# 5. Scope

## Included in Milestone 1

### Dataset

- MRI dataset loading
- Dataset indexing
- Patient/study identification
- Metadata inspection
- Dataset statistics

### Preprocessing

- Orientation handling
- Resampling where required
- Cropping
- Padding
- Resizing
- Intensity normalization
- Missing/corrupt file detection
- Train/validation/test splitting

### Visualization

- Axial slices
- Sagittal slices
- Coronal slices
- Slice navigation
- Volume dimensions
- Intensity histogram
- Before/after preprocessing comparison
- Basic 3D volume visualization

### Research utilities

- Dataset report
- Processing logs
- Reproducible configuration
- Data validation
- Export processed tensors

---

# 6. Explicitly NOT Included

Do NOT implement these in Milestone 1:

- Diffusion model training
- VQ-GAN training
- VQ-VAE training
- Disease diagnosis
- Cancer prediction
- Medical recommendations
- Clinical decision support
- Patient diagnosis
- LLM chatbot
- Medical report generation
- Automatic treatment recommendations
- Cloud deployment
- Federated learning
- Patient-identifying information

The application must remain a research/data-processing system.

---

# 7. Dataset Strategy

The codebase must support publicly available 3D MRI datasets.

The initial implementation should support a dataset organized approximately as:

```text
data/
├── raw/
│   ├── patient_001/
│   │   └── scan.nii.gz
│   ├── patient_002/
│   │   └── scan.nii.gz
│   └── ...
│
├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
│
└── metadata/
```

The implementation must NOT hard-code patient IDs or assume a specific number of scans.

Dataset paths must be configurable.

Example:

```yaml
dataset:
  name: "MRI"
  raw_dir: "./data/raw"
  processed_dir: "./data/processed"
```

---

# 8. Supported MRI Formats

The system should prioritize:

```text
.nii
.nii.gz
```

Optional support:

```text
.dcm
```

If DICOM support is implemented, conversion should happen during preprocessing rather than throughout the ML pipeline.

Internally, the ML pipeline should work with standardized numerical tensors.

---

# 9. Core Data Pipeline

The pipeline should be:

```text
Raw MRI
   ↓
File validation
   ↓
Load volume
   ↓
Orientation normalization
   ↓
Resampling
   ↓
Cropping / Padding
   ↓
Resize
   ↓
Intensity normalization
   ↓
Tensor conversion
   ↓
Dataset split
   ↓
Save processed volume
```

Every transformation must be deterministic when a random seed is provided.

---

# 10. MRI Loading

Create a reusable MRI loader.

Example conceptual API:

```python
volume = load_mri(path)
```

The loader should return:

```text
volume
metadata
```

The volume should be represented as a NumPy array initially.

Example:

```text
numpy.ndarray
shape = (H, W, D)
```

The loader must validate:

- file exists
- file can be opened
- volume has valid dimensions
- volume contains finite values
- volume is not empty
- voxel spacing is available where applicable

---

# 11. Metadata Extraction

For every scan collect available metadata such as:

```text
file_path
volume_shape
voxel_spacing
orientation
data_type
minimum_intensity
maximum_intensity
mean_intensity
standard_deviation
```

Do not expose personally identifying patient information in the UI.

---

# 12. Preprocessing Requirements

## 12.1 Orientation

MRI volumes should be converted into a consistent orientation where the dataset format permits it.

The application should record the original orientation and resulting orientation.

---

## 12.2 Resampling

If voxel spacing differs substantially between scans, resample to a consistent spacing.

The target spacing must be configurable.

Example:

```yaml
preprocessing:
  target_spacing:
    - 1.0
    - 1.0
    - 1.0
```

Do not assume this exact spacing is optimal for every MRI dataset.

---

## 12.3 Cropping

The system should support center cropping.

Example:

```yaml
preprocessing:
  crop_size:
    - 256
    - 256
    - 32
```

The implementation must gracefully handle volumes smaller than the requested crop size.

---

## 12.4 Padding

If the MRI is smaller than the target dimensions, pad it to the configured shape.

Padding strategy should be configurable.

Default:

```text
constant padding
```

---

## 12.5 Resizing

After cropping/padding, resize the volume to the target shape.

Example:

```text
256 × 256 × 32
```

The target shape must be configurable.

---

# 13. Intensity Normalization

MRI intensity values can vary between scans.

Implement configurable normalization.

Primary Milestone 1 method:

```text
Min-Max normalization
```

Mapping:

```text
original intensity
        ↓
normalized intensity
        ↓
[-1, 1]
```

This follows the general preprocessing approach described in the Medical Diffusion paper, where datasets were normalized to the range -1 to 1.

Also allow optional:

```text
Z-score normalization
```

but do not make it the default.

---

# 14. Train / Validation / Test Split

Create a reproducible dataset split.

Default:

```text
Train       70%
Validation  15%
Test        15%
```

The split ratio must be configurable.

Use a fixed random seed.

Most importantly:

**Do not allow scans from the same patient to appear in multiple splits.**

The split should happen at the patient/study level when patient identifiers are available.

---

# 15. Data Validation

Before a scan is added to the processed dataset, run validation.

Checks:

```text
✓ File exists
✓ Volume readable
✓ Correct dimensionality
✓ No NaN values
✓ No infinite values
✓ Non-zero volume
✓ Valid intensity range
✓ Target dimensions achievable
```

Invalid scans should be recorded in:

```text
logs/validation_errors.csv
```

The pipeline should continue processing valid scans and record invalid sources separately.

---

# 16. Visualization Dashboard

Build a simple **Streamlit research dashboard**.

The dashboard should have these sections:

```text
MRI Research Dashboard

├── Dataset Overview
├── MRI Viewer
├── Preprocessing
├── Intensity Analysis
├── Dataset Quality
└── Export
```

---

# 17. Dataset Overview Page

Display:

```text
Total scans
Training scans
Validation scans
Test scans

Average volume size
Minimum volume size
Maximum volume size

Average voxel spacing
```

Example:

```text
Total MRI scans       998
Training              699
Validation            149
Testing               150

Average shape         240 × 240 × 32
```

Numbers must be calculated dynamically from the actual dataset.

Do not hard-code results.

---

# 18. MRI Viewer

The MRI viewer is a core feature.

Allow the user to select a scan.

Display:

```text
Axial
Sagittal
Coronal
```

views.

Provide slice sliders:

```text
Axial slice:     [────●────]
Sagittal slice:  [────●────]
Coronal slice:   [────●────]
```

Display the selected slice.

Also show:

```text
Current slice
Total slices
Volume dimensions
```

---

# 19. Before vs After Viewer

Provide:

```text
Original MRI     |     Processed MRI
```

The researcher should be able to inspect aligned source and processed MRI slices.

Include:

- same slice position where possible
- synchronized navigation
- dimensions
- intensity range

---

# 20. Intensity Histogram

For a selected MRI volume, display its intensity distribution.

Example:

```text
Intensity
   │
   │       ███
   │     ███████
   │   ███████████
   │ ███████████████
   └──────────────────
```

Display:

```text
Minimum
Maximum
Mean
Median
Standard deviation
```

Allow comparison:

```text
Before normalization
vs
After normalization
```

---

# 21. 3D Visualization

Provide an optional basic 3D visualization of the volume.

The rendering scope is local research visualization.

It should allow the researcher to understand the volumetric structure.

Possible implementation:

```text
Plotly
```

or another lightweight visualization library.

If 3D rendering is computationally expensive, provide it as an optional feature.

---

# 22. Dataset Quality Report

Create an automated report containing:

```text
Dataset Summary
----------------
Number of scans
Number of valid scans
Number of invalid scans

Shape Statistics
----------------
Min
Max
Mean

Intensity Statistics
--------------------
Min
Max
Mean
Std

Spacing Statistics
------------------
Min
Max
Mean
```

Also report:

```text
Missing files
Corrupted files
NaN-containing volumes
Dimension inconsistencies
```

---

# 23. Export

The user should be able to export processed data into:

```text
.npy
```

or:

```text
.pt
```

Recommended structure:

```text
data/
└── processed/
    ├── train/
    │   ├── sample_0001.npy
    │   ├── sample_0002.npy
    │   └── ...
    ├── validation/
    └── test/
```

Also create:

```text
metadata.csv
```

containing processing information.

---

# 24. Configuration System

Do not hard-code preprocessing parameters.

Create:

```text
configs/
└── config.yaml
```

Example:

```yaml
dataset:
  raw_dir: "./data/raw"
  processed_dir: "./data/processed"

preprocessing:
  target_shape: [256, 256, 32]
  normalization: "minmax"
  normalization_range: [-1, 1]

split:
  train: 0.70
  validation: 0.15
  test: 0.15

seed: 42
```

The application should read these values dynamically.

---

# 25. Project Structure

Create the following structure:

```text
medical-diffusion-mri/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── preprocessing.py
│   │   ├── validation.py
│   │   └── splitting.py
│   │
│   ├── visualization/
│   │   ├── mri_viewer.py
│   │   ├── plots.py
│   │   └── volume_viewer.py
│   │
│   └── utils/
│       ├── config.py
│       ├── logging.py
│       └── seed.py
│
├── scripts/
│   ├── preprocess.py
│   ├── validate_dataset.py
│   └── generate_report.py
│
├── app/
│   └── streamlit_app.py
│
├── tests/
│   ├── test_loader.py
│   ├── test_preprocessing.py
│   └── test_validation.py
│
└── logs/
```

---

# 26. Technology Stack

Use:

### Programming

```text
Python 3.10+
```

### Machine Learning

```text
PyTorch
NumPy
SciPy
```

### Medical Imaging

Prefer:

```text
MONAI
NiBabel
SimpleITK
```

### Visualization

```text
Matplotlib
Plotly
Streamlit
```

### Data

```text
Pandas
```

Do not introduce unnecessary frameworks.

---

# 27. Coding Requirements

Code must be:

- simple
- readable
- modular
- reproducible
- type-safe where useful
- documented
- easy for a student/researcher to understand

Avoid:

- unnecessary abstractions
- complex class hierarchies
- over-engineering
- hard-coded dataset values
- hidden preprocessing
- huge monolithic files

Every important preprocessing operation should be independently testable.

---

# 28. Testing Requirements

Create unit tests for:

### MRI loading

```text
valid file → loads successfully
invalid file → handled gracefully
```

### Normalization

Verify:

```text
output minimum ≈ -1
output maximum ≈ 1
```

for Min-Max normalization.

### Shape

Verify:

```text
processed_volume.shape == target_shape
```

### Validation

Test:

```text
NaN volume
empty volume
wrong dimensions
missing file
```

### Dataset split

Verify that patient IDs do not overlap between:

```text
train
validation
test
```

---

# 29. Reproducibility

Every experiment must be reproducible.

Set seeds for:

```text
Python
NumPy
PyTorch
```

Store:

```text
configuration
random seed
dataset version
preprocessing parameters
```

in the experiment metadata.

---

# 30. Research Metrics

Milestone 1 does not need to train a diffusion model, but establish the metrics that will later be used.

Prepare support for:

### Image quality

```text
PSNR
SSIM
MS-SSIM
```

### Reconstruction

```text
MSE
MAE
```

### Segmentation

Future Milestones:

```text
Dice
IoU
```

The Medical Diffusion paper used MS-SSIM to compare image diversity and Dice score to evaluate downstream breast segmentation performance.

---

# 31. Future Architecture

The codebase must be designed so that Milestone 2 can add:

```text
                 Processed MRI
                       │
                       ▼
                  VQ-VAE/GAN
                       │
                       ▼
                Latent Dataset
                       │
                       ▼
                3D Diffusion
                       │
                       ▼
              Generated Latent
                       │
                       ▼
                  Decoder
                       │
                       ▼
               Synthetic MRI
```

Do not implement this yet.

Create clean interfaces so the future models can consume:

```text
torch.Tensor
shape:
[B, C, H, W, D]
```

---

# 32. Future Research Direction

The eventual research goal should go beyond simply reproducing Medical Diffusion.

Potential future direction:

**Longitudinal MRI generation**

```text
MRI at T0
   +
Patient/clinical features
   ↓
Generative model
   ↓
Predicted MRI at T1
   ↓
Actual MRI at T1
   ↓
Quantitative comparison
```

This should remain outside Milestone 1 but the architecture should not prevent adding it later.

---

# 33. UI Design

The interface should be minimal and research-oriented.

Sidebar:

```text
Medical Diffusion Research

Dataset
MRI Viewer
Preprocessing
Statistics
Quality Report
```

Main dashboard:

```text
┌───────────────────────────────────────────┐
│        3D MRI Research Dashboard          │
├───────────┬───────────┬───────────────────┤
│ Scans     │ Train     │ Test              │
│ 998       │ 699       │ 150               │
├───────────┴───────────┴───────────────────┤
│                                           │
│             MRI Viewer                    │
│                                           │
│       [Axial] [Sagittal] [Coronal]       │
│                                           │
│              MRI IMAGE                   │
│                                           │
├───────────────────────────────────────────┤
│ Intensity Distribution                   │
│              Histogram                   │
└───────────────────────────────────────────┘
```

Do not make the UI look like a consumer healthcare application.

It should look like a **scientific research dashboard**.

---

# 34. Error Handling

The application must not crash because of one malformed MRI file.

Instead:

```text
WARNING:
Unable to process scan_023.nii.gz

Reason:
Invalid volume dimensions
```

Record the error in logs and continue processing the remaining dataset.

---

# 35. Privacy & Safety

The project must be clearly labeled:

Place the project's research-use note at the end of the page.

Do not display:

- patient names
- addresses
- phone numbers
- hospital IDs
- other unnecessary identifying information

When possible, anonymize metadata before processing.

---

# 36. Milestone 1 Acceptance Criteria

Milestone 1 is complete only when all of the following are true:

### Dataset

- [ ] MRI dataset can be loaded.
- [ ] Dataset can contain variable numbers of scans.
- [ ] Invalid scans are detected.
- [ ] Dataset statistics are automatically calculated.

### Preprocessing

- [ ] MRI orientation can be standardized.
- [ ] MRI volumes can be resampled.
- [ ] MRI volumes can be cropped/padded.
- [ ] MRI volumes can be resized.
- [ ] MRI intensities can be normalized to [-1, 1].
- [ ] Processing parameters are configurable.

### Visualization

- [ ] Axial view works.
- [ ] Sagittal view works.
- [ ] Coronal view works.
- [ ] Slice navigation works.
- [ ] Original vs processed comparison works.
- [ ] Intensity histogram works.
- [ ] Basic 3D visualization works or is clearly documented if disabled for performance.

### Dataset splitting

- [ ] Train/validation/test split works.
- [ ] Split is reproducible.
- [ ] Patient-level leakage is prevented where identifiers exist.

### Research

- [ ] PSNR/SSIM utilities exist.
- [ ] Dataset quality report can be generated.
- [ ] Processing metadata is saved.

### Engineering

- [ ] Code is modular.
- [ ] Tests pass.
- [ ] README explains setup.
- [ ] Application launches successfully.
- [ ] No hard-coded patient data.
- [ ] Configuration is externalized.

---

# 37. Expected Final Demo

At the end of Milestone 1, the researcher should be able to launch:

```bash
streamlit run app/streamlit_app.py
```

Then:

```text
1. Select MRI dataset
        ↓
2. View dataset statistics
        ↓
3. Select MRI scan
        ↓
4. Explore 3D slices
        ↓
5. Run preprocessing
        ↓
6. Compare original/processed volume
        ↓
7. View intensity histogram
        ↓
8. Validate dataset
        ↓
9. Export processed dataset
        ↓
10. Generate research report
```

---

# 38. Final Deliverables

Milestone 1 must produce:

```text
✓ Working MRI preprocessing pipeline
✓ Working 3D MRI viewer
✓ Dataset validation system
✓ Dataset statistics dashboard
✓ Configurable preprocessing
✓ Train/validation/test split
✓ Processed MRI dataset
✓ Dataset quality report
✓ Unit tests
✓ README
✓ Research documentation
```

The most important outcome is:

> **A clean standardized 3D MRI dataset that is ready to be consumed by the VQ-VAE/VQ-GAN and diffusion models in the next milestone.**

---

# 39. Cursor Implementation Instructions

Implement this PRD incrementally.

First inspect the repository.

Then create the project structure.

Do NOT implement the diffusion model yet.

Start with:

1. Configuration system.
2. MRI loader.
3. Dataset validator.
4. Preprocessing pipeline.
5. Dataset splitter.
6. Visualization utilities.
7. Streamlit dashboard.
8. Dataset report.
9. Tests.
10. README.

After each major component:

- run tests
- fix errors
- verify imports
- verify the application launches

Use actual MRI inputs and record model outputs and experiment settings accurately.

If no MRI dataset is available locally, implement the pipeline against a small synthetic 3D NumPy volume for testing, but clearly label it as test data and keep the real MRI dataset integration ready.

Do not download datasets automatically without explicit configuration.

---

# 40. Definition of Done

Milestone 1 is considered successful when a researcher can take a publicly available MRI dataset, point the configuration to it, run:

```bash
python scripts/validate_dataset.py
python scripts/preprocess.py
python scripts/generate_report.py
streamlit run app/streamlit_app.py
```

and obtain:

```text
Raw MRI
   ↓
Validated MRI
   ↓
Standardized 3D MRI
   ↓
Visualization
   ↓
Statistics
   ↓
Quality Report
   ↓
Processed Dataset
```

which is ready for **Milestone 2: latent representation learning using VQ-VAE/VQ-GAN**.

Research project only. Not for real medical use.
