# Full application and UI verification

**Verified:** 21 October 2026, Asia/Kolkata.

## Results

| Check | Result |
|---|---|
| Full automated suite, including real MRI and trained checkpoints | **61 passed in 36.04 seconds; no skips** |
| Chrome end-to-end workflow scenarios | **20 passed** |
| Downloaded files independently decoded/validated | **15 files passed** |
| Actual NIfTI CLI validation, preprocessing, reporting | **3 valid real scans; 0 rejected** |
| Research/download/training CLI argument entry points | **13 passed** |
| Dependency consistency (`python -m pip check`) | **No broken requirements** |
| Live Streamlit health | **HTTP 200, `ok`** |
| Browser JavaScript exceptions, console errors, failed local requests | **0 observed** |

Environment: Python 3.12, PyTorch 2.14.1+cu130, CUDA 13.0, RTX 5070 Ti Laptop
GPU. Browser checks used installed **Chrome 154.0.8037.100** in headless mode
with **Playwright 1.64.1**. WebGL chart rendering used software rendering in
the test browser; model inference executed through the application's trained
PyTorch checkpoints.

## Browser workflows exercised

1. Fresh launch and all eight workspace navigation entries.
2. Missing configuration file, user-facing error, and recovery.
3. Missing raw-data directory, user-facing error, and successful re-indexing.
4. All three local NIfTI scans; first/last axial, sagittal, and coronal slices.
5. Optional 3D volume rendering, with a live canvas and no more than 48³ values.
6. Preprocessing comparison in all three planes, synchronized navigation,
   interpolation metrics, tensor/metadata/configuration downloads.
7. Z-score/min-max normalization, all three padding modes, crop/resize controls,
   and optional spacing resampling.
8. Excessive allocation rejected gracefully, followed by a successful preview.
9. Intensity analysis, rendered histograms, and calculated interpolation baseline.
10. Dataset-quality report download with actual scan/error counts.
11. Real three-scan batch export and all export downloads.
12. MRNet official train/validation browsing and each acquired plane's slice bounds.
13. Real MRNet subset preparation, existing-output rejection, and missing-folder recovery.
14. Trained-model dashboard metrics and computed-evidence download.
15. All eight saved generated volumes and each tensor slice direction/boundary.
16. First/middle/last validation reconstruction pairs and live trained reconstruction.
17. Fresh-seed live generation, downloaded output, and byte-identical repeated-seed output.
18. Generation-diagnostics download and the explicit missing-downstream-results state.
19. Tablet/phone navigation and layout checks on overview, viewer, preprocessing,
    MRNet, and model-experiment pages.
20. Browser exception, console, connection, and failed-request checks.

Desktop workflows were tested at **1440 px**. Responsive checks used **900 px**
and **390 px**, including sidebar opening/closing. The checked layouts had no
horizontal page overflow, and visible plots remained at least 250 px wide.
These are browser viewport checks, not physical-device or cross-browser certification.

## Independent download and export checks

- Preview NumPy tensors decoded as finite float32 `[1,128,128,128]` arrays.
  Min-max outputs reached [-1,1]; z-score mean/std matched approximately 0/1.
- Preview metadata retained a finite RAS affine and the requested output shape.
- Downloaded preview YAML loaded successfully through the application config parser.
- The real NIfTI export contained three checksum-matching tensors and source
  checksums matching the recorded Slicer inputs.
- MRNet UI preparation exported four valid `[1,32,32,16]` tensors, split 2/2
  without cross-split exam overlap.
- Saved generated-volume downloads matched their recorded SHA-256 values.
- New live-generated tensors were finite, normalized `[1,128,128,32]` arrays;
  repeated seed 2045 reproduced identical bytes and differed from the saved sample.
- Paired reconstruction reports contained 24 validation results; model evidence
  reported the actual 96/24 cohort and 40/300 completed training epochs.

Browser screenshots, the browser report, downloaded test artifacts, and JUnit
output are retained locally outside the tracked source tree.

## Defects found and corrected

| Defect | Correction |
|---|---|
| Valid YAML dimensions/crops above the preview's default 1024 limit crashed the page | Preview bounds accommodate the validated configuration |
| Valid spacing below 0.1 mm or above 20 mm crashed number inputs | Spacing controls accommodate configured values |
| Malformed research run JSON could crash experiment discovery | Invalid metadata objects, epoch values, and checkpoint names are skipped |
| Failed re-indexing could retain an old dataset, preview, and export | Previous snapshot is cleared before each indexing attempt |
| Three-column layouts cramped MRI views on tablet widths | Main-area columns stack on narrower screens |
| Expanded raw reconstruction JSON pushed controls far down the page | Raw report starts collapsed |
| MRNet reconstruction plots used an anatomical plane label without an affine | Plots are labeled as acquired-series slices in tensor coordinates |

Eleven regression cases were added to the automated suite for the configuration,
index-state, and malformed-metadata failures. Additional probes exercised empty
and corrupt datasets and preview invalidation after parameter/dataset changes.

## Re-run

```bash
python -m pytest -q -ra
python -m pip check
python -m streamlit run app/streamlit_app.py
```

Downloaded real samples and completed M4 model artifacts are required for the
real-data/checkpoint cases. Without them those cases skip explicitly. See
[the README](../README.md) for acquisition/training commands and
[the M4 demo guide](milestones/m4-demo.md) for interactive inference.

These checks establish the tested software and UI behavior. The generated-volume
anatomical-quality limitations recorded in the [baseline results](results/m4-baseline.md)
are unchanged by UI verification.
