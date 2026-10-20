# MRI data-pipeline reference

## Input contracts

### NIfTI

Use `.nii` or `.nii.gz` files grouped by subject beneath `raw_dir`. The first
directory identifies the subject and must connect repeat scans and timepoints.
Flat datasets require a CSV manifest:

```csv
file_path,patient_id
session_a/scan.nii.gz,anonymous-subject-1
session_b/scan.nii.gz,anonymous-subject-1
session_c/scan.nii.gz,anonymous-subject-2
```

Manifest paths are relative to `raw_dir`, must remain inside it, and may identify
missing files for validation. Duplicate files are rejected. The pipeline cannot
infer that different identifiers belong to one person.

### MRNet

Released arrays are `[slice,row,column]`, with no reliable patient-space affine
or physical voxel spacing. Each plane is a separate acquired series. The adapter
preserves the official train/validation assignment, relying on Stanford's
published patient-disjoint split policy. It does not fabricate patient identities
or register the acquired planes.

## Configuration

Relative paths resolve against the YAML file, independent of the shell's working
directory. Use `--config path/to/config.yaml` for the NIfTI CLI scripts. Local
overrides can use ignored filenames such as `configs/config.local.yaml`.

The default NIfTI configuration uses 2 mm spacing, no center crop, 128³ output,
and min-max normalization to [-1,1]. The original technical PRD example is in
`configs/prd-example.yaml`. Model experiments have separate training configurations.

Dashboard preview edits affect the preview. Download and select its YAML to use
those parameters for batch export. Re-index after source files change.

## NIfTI preprocessing

1. Validate readability, exactly three dimensions, finite nonconstant intensities,
   positive spacing, nonsingular affine, and voxel-allocation limits.
2. Reorient to canonical RAS. Oblique geometry is retained if spacing resampling
   is disabled; enabled resampling creates an axis-aligned RAS output grid.
3. Resample with linear interpolation. Requested spacing is in millimeters;
   meter/micron affines are converted. Unknown units require correcting the source
   or disabling spacing resampling.
4. Optionally center crop/pad. Constant, edge, and reflect padding are supported.
   Constant padding uses zero in source intensity units. The affine origin changes
   to preserve voxel-to-world mapping.
5. Resize to the requested shape, preserving first/last voxel centers on
   nonsingleton axes. This changes effective spacing, recorded in the output affine.
6. Normalize the complete volume, including padding, using min-max or z-score.
7. Export channel-first float32 arrays and transformation metadata.

Cropping can discard anatomy. No registration, skull stripping, bias correction,
or modality selection happens implicitly. Four-dimensional inputs must be separated
explicitly before ingestion. DICOM ingestion is outside this implementation.

## Tensor layouts

| Source | Stored sample | DataLoader batch |
|---|---|---|
| Canonical NIfTI | `[1,R,A,S]` | `[B,1,R,A,S]` |
| MRNet selected series | `[1,row,column,slice]` | `[B,1,row,column,slice]` |

```python
from torch.utils.data import DataLoader
from src.data.dataset import MRIDataset

dataset = MRIDataset('data/processed/<run-id>', split='train')
loader = DataLoader(dataset, batch_size=2, shuffle=False)
batch = next(iter(loader))
```

MRNet training transposes, resizes with trilinear interpolation, and normalizes
the selected acquired series. The models treat the three spatial dimensions as
generic tensor axes, without assigning an invented NIfTI affine.

## Splits and export

Generic NIfTI splits use seeded subject shuffling and largest-remainder allocation
(70/15/15 by default). Ratios apply to subjects, so scan proportions can differ.
Small datasets can have empty splits. Adding subjects can change assignments.
MRNet instead retains its official assignments.

Each NIfTI export creates a fresh run directory with:

```text
train/  validation/  test/
metadata.csv
experiment.json
report.json
validation_errors.csv
processing_log.csv
```

`metadata.csv` records source/tensor checksums, pseudonymous IDs, split assignments,
and original/processed geometry. `experiment.json` records resolved configuration,
seed, dependency versions, and assignments. Set `export_format: pt` for PyTorch
files instead of NumPy. A bad scan is logged while valid scans continue.

Consume completed exports with final metadata/report files. Interrupted processes
can leave partial run directories. Runtime metadata can contain local paths and
belongs in ignored output directories.

## Metrics and visualization

Paired reconstruction metrics require matching shape, physical alignment, and
intensity scale. Default min-max tensors use `data_range=2`. Identical inputs have
infinite PSNR. Volumetric SSIM adapts its window for small volumes.

The dashboard's interpolation baseline downsamples a processed volume to half
resolution, then reconstructs it on the same grid. Original/processed comparisons
sample original intensities on the output world grid; they are not reconstruction
metrics across different intensity scales.

Generation uses a separate adaptive-scale 3D MS-SSIM protocol with its window and
scale count recorded. The optional `pytorch-msssim` wrapper uses its upstream
five-scale constraints. Neither metric establishes anatomical validity.

The NIfTI viewer displays canonical slices with physical aspect ratios. Optional
3D rendering sends at most 48³ sampled values to the local browser. Source volumes
are processed one at a time, but previews can hold multiple arrays; `max_voxels`
bounds individual volumes rather than total process memory.
