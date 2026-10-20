# Milestone 1 research notes

## Scope delivered

This implementation turns the PRD into a reusable data layer and local research
application. NiBabel and SciPy provide the necessary image operations without
adding MONAI or SimpleITK for overlapping functionality. Plotly provides slice,
histogram, and basic volume views. PyTorch is the tensor export/loading interface.

The original technical Milestone 1 delivery concerned data handling. Modeling
was subsequently added under `src/models/` and `src/training/`; the current
college assessment deliverables are in [milestones/README.md](milestones/README.md).
Artificial arrays are clearly labeled temporary pytest fixtures. Three real
Slicer samples support NIfTI integration checks, while a separate real MRNet
subset supplies the knee-model experiment. The initial NIfTI application asks
the researcher to index their configured dataset.

## Spatial conventions

NIfTI affine matrices map voxel centers into world coordinates. Canonicalization
permutes/flips array axes without changing that physical mapping. Cropping and
padding translate the affine origin; resizing updates basis vectors. Spacing
resampling is distinct from shape resizing, and the latter can change the final
spacing. All transformations are recorded in export metadata.

Slice display uses increasing R/A/S axis labels and physical aspect ratios. It
is a research visualization, not a radiological display convention. Oblique
scans with resampling disabled use nearest canonical axis labels; their planes
remain oblique in world space. The synchronized comparison samples the raw
scan on the processed grid so both panels refer to the same physical locations.
Always review the uncropped source separately.

## Dataset integrity

Subject grouping is an explicit data contract. Source layout or manifest
identifiers must connect repeat scans and all timepoints from each person.
The software cannot detect that two different identifiers belong to one person.
No study-level fallback overrides missing subject identity. Subject hashes are
stable across runs, but are not protection against guessing known identifiers.

Invalid source files and failed transformations are excluded from tensors and
recorded by pseudonymous scan ID. Reports count preprocessing failures as invalid
in an export report; standalone validation reports cover source validity only.
Shape and spacing summaries are per axis; intensity mean/std summaries are
per scan rather than a voxel-weighted global estimate.

## Future model boundary

`MRIDataset` returns channel-first float32 tensors. Future encoders should read
the completed run metadata and explicitly adopt the same axis convention,
normalization, shape, and split. Preserve this frozen dataset version when
comparing models. Do not create independent slice-level random splits.

Longitudinal extensions can add timepoint and modality mappings in a separate
de-identified manifest, retaining subject-level grouping. Spatial registration
and appropriate evaluation would be separate, explicit processing steps.

## Real-data acceptance still needed

Run validation on a licensed dataset, check its subject mapping, confirm units,
inspect anatomy before and after cropping, and review the run error log. The
shape and spacing are configurable; the active Slicer configuration uses whole
field-of-view 128³ exports rather than the original PRD crop. This repository
does not claim that numerical fixtures establish real-world MRI quality.
