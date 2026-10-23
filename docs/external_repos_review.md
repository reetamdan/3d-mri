# External implementation review — constrained knee MRI research

Research dates: 23–24 October 2026. Repository metadata, release dates, branches,
licenses and source were inspected with `gh repo view`, `gh api`, `gh release
list`, `gh search code/issues` and shallow `gh repo clone` checkouts outside this
repository. Decisions below distinguish importable software from architectural
ideas and unavailable evidence. No reference checkout is vendored here.

## Versions and repository state

| Repository | Inspected default-branch revision | Last commit date | State / release observation |
|---|---|---|---|
| [MONAI](https://github.com/Project-MONAI/MONAI) | `6920610c310200d683a3b00987a05893ccf1ca5f` | 2026-10-22 | Maintained; use stable 1.6.1 (2026-09-27), tag `f2ec87ae9b536baaeca1d6bcc2b6349369e2af8f` |
| [MONAI tutorials](https://github.com/Project-MONAI/tutorials) | `d7c326e2b5b72423a082c9cee3e41035c4c6a42a` | 2026-10-20 | Maintained; no release required |
| [TorchIO](https://github.com/TorchIO-project/torchio) | `9b1be95027567b8587b05f48c22122b1dcac6cc0` | 2026-10-16 UTC | Main is a redesigned 2.0 prerelease; use stable 1.2.1, tag `d433fa7fef9f8507ca8a71433f1636eadbd02fcf` (2026-06-02) |
| [SKM-TEA](https://github.com/StanfordMIMI/skm-tea) | `6cb8460c8a0d3f29ce8db0cae7e4d774c2a2716a` | 2022-12-11 | Not archived, but legacy training dependencies |
| [fastMRI](https://github.com/facebookresearch/fastMRI) | `91f2df4711adbb6d643df1810f234e4abcf5881b` | 2025-01-21 | **Archived**; latest release v0.3.0 (2023-06-26) |
| [Min-SNR](https://github.com/TiankaiHang/Min-SNR-Diffusion-Training) | `5189997d6bbde5a1301bcf31a54d2b3589f7309c` | 2024-12-10 | Reference implementation; no clear top-level software license |
| [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch) | `4103bf2c7be993b6e0ac21d88684818c9880c478` | 2026-09-24 UTC | Maintained; source metadata 1.31.6 differs from latest GitHub release 1.27.21 |
| [3D MedDiffusion](https://github.com/ShanghaiTech-IMPACT/3D-MedDiffusion) | `d1beb8f8d5d81edc0358269682edb58aa9ad59ea` | 2026-01-28 | Research source; no clear top-level license |
| [TCAM-Diff](https://github.com/Fredy-Zhang/TCAM-Diff) | `c12d77800b3b07ffd706992573c3aef928d9cdd4` | 2026-04-08 | **Current main contains only README, LICENSE and environment.yml; model/config source advertised by README is absent** |
| [MedicalNet](https://github.com/Tencent/MedicalNet) | `20f76aaab5cac8056eaf50b79ed97c09dbfbd3bd` | 2025-11-27 | Legacy code; public pretrained models referenced |
| [EDM](https://github.com/NVlabs/edm) | `008a4e5316c8e3bfe61a62f874bddba254295afb` | 2024-03-16 | Reference implementation; not a knee-MRI model |

An additional, narrowly scoped source review after the one-volume diffusion
failure inspected [Uber CoordConv](https://github.com/uber-research/CoordConv) at
`27fab8b86efac87c262c7c596a0c384b83c9d806`; it was not part of the original eleven
requested repositories.

`Project-MONAI/GenerativeModels` is the historical route; current `monai.networks`
and `monai.inferers` are preferred. Dates describe inspected commits, not promises
of active support. GitHub push timestamps can differ from commit timestamps.

## MONAI and tutorials

- **Purpose:** maintained 3D KL/VQ models, diffusion components, losses and inferers.
- **Relevant source inspected:**
  [`monai/networks/nets/autoencoderkl.py`](https://github.com/Project-MONAI/MONAI/blob/6920610c310200d683a3b00987a05893ccf1ca5f/monai/networks/nets/autoencoderkl.py)
  (`AutoencoderKL`, `Encoder`, `Decoder`, `AEKLDownsample`);
  `monai/networks/blocks/upsample.py`; `diffusion_model_unet.py`
  (`DiffusionModelUNet`, context-conditioned spatial transformers);
  `monai/networks/schedulers/ddpm.py` (`DDPMScheduler.step`);
  `monai/inferers/inferer.py` (`LatentDiffusionInferer`);
  `monai/losses/perceptual.py` (`PerceptualLoss`, `MedicalNetPerceptualSimilarity`);
  `monai/apps/generation/maisi/networks/autoencoderkl_maisi.py`;
  [tutorial `generation/3d_ldm/3d_ldm_tutorial.ipynb`](https://github.com/Project-MONAI/tutorials/blob/d7c326e2b5b72423a082c9cee3e41035c4c6a42a/generation/3d_ldm/3d_ldm_tutorial.ipynb).
- **Useful implementation:** posterior mean/sigma, log-variance clamping,
  deterministic `reconstruct` versus sampled training forward; optional activation
  checkpointing; latent scaling applied before diffusion and inverted before decode.
  Tutorial uses batch-mean **summed** KL with weight 1e-6, L1, and optional perceptual/GAN terms.
- **Hardware assumptions:** tutorial channels 32/64/64 and attention are much
  heavier than needed here. MAISI targets large volumes and includes CPU transfers
  and split operations; it is not a drop-in lightweight knee autoencoder.
- **License / direct use:** Apache-2.0; import official MONAI 1.6.1 rather than copy.
- **Adaptation:** compact channels 8/16/32, 8 latent channels, no attention/GAN in
  the first comparison. Native KL downsampling is hard-coded isotropic. Confirmed
  by [issue #8447](https://github.com/Project-MONAI/MONAI/issues/8447). Adapt only
  padding/strides and mirrored interpolation scale factors for 4×4×2, preserving
  residual/posterior/decoder modules and parameter initialization.
- **Decision/reason:** first KL comparison is directly imported; anisotropy is a
  small shape-tested adapter. Preserve the existing validated denoiser until the
  representation and tiny-data gates justify further diffusion work. MedicalNet
  perceptual use requires explicit weights/provenance; no implicit startup download.

## TorchIO

- **Purpose:** conventional MRI augmentation and patch sampling.
- **Source inspected at stable 1.2.1:** `src/torchio/transforms/augmentation/spatial/random_affine.py`
  (`RandomAffine`, `Affine`); `random_anisotropy.py`; intensity
  `random_noise.py`, `random_bias_field.py`, `random_motion.py`,
  `random_ghosting.py`, `random_spike.py`; `src/torchio/data/sampler/uniform.py`.
  Main's newer `transforms/intensity/bias_field.py` is a different GPU-native API;
  it must not be mixed with the stable Random* API.
- **Useful implementation:** parameter sampling is driven by Torch RNG; affine
  translation is physical millimeters; intensity artifacts simulate specific
  acquisition failures; uniform patch selection is voxel-grid based.
- **Hardware assumptions:** stable spatial transforms use SimpleITK/CPU; cap
  augmentation threads and use modest transforms rather than a heavy artifact chain.
- **License / direct use:** Apache-2.0; import stable package.
- **Adaptation:** MRNet has no reliable physical geometry. Use a documented
  computational unit grid for small row/column transforms, preserve slice order,
  avoid left/right flips and invented mm claims. Apply weak noise/bias to [0,1]
  intensities before returning to [-1,1]; multiplying negative normalized
  intensities by a bias field is not the desired MRI intensity operation.
- **Decision/reason:** conventional augmentation is an essential independent arm.
  Motion/ghosting/spikes/aggressive anisotropy are researched but excluded from
  initial defaults because they can erase the small structures being evaluated.

## SKM-TEA

- **Purpose:** tissue/pathology annotations and anatomy-aware evaluation concepts.
- **Source inspected:** `DATASET.md`, `MODEL_ZOO.md`,
  `skm_tea/data/dataset.py`, `skm_tea/modeling/monai_nets.py`,
  `skm_tea/evaluation/qdess_evaluation.py` (`SkmTeaEvaluator`).
- **Useful implementation:** scan-level aggregation, normalization reversal,
  separate reconstruction/segmentation/qMRI tasks; explicit subject/timepoint IDs,
  voxel spacing, orientation and tissue classes. DICOM HDF5 `echo1`/`echo2` are
  `[X,Y,Z]`; `seg` is `[X,Y,Z,6]`, including medial/lateral tissues.
- **Hardware/data assumptions:** qDESS is not MRNet's acquired contrast. Full data
  documentation quotes 900 GB compressed / 1.6 TB extracted and account-based access.
  Published model wrappers are 2D and rely on legacy Meddlr/Lightning dependencies.
- **License / direct use:** software MIT; dataset access/use terms are separate.
- **Adaptation:** support explicit compatible volume/annotation manifests, frozen
  caller-supplied anatomy models, and scan/patient aggregation. Never transplant
  qDESS segmentations into MRNet coordinates or call generic resized boxes ACL ROIs.
- **Decision/reason:** anatomy-proxy infrastructure is appropriate; actual Dice,
  tissue-volume or local-lesion results require suitable images/masks/weights and
  cannot be manufactured from the currently available exam labels.

## fastMRI

- **Purpose:** larger unlabeled knee representation-pretraining data.
- **Source inspected:** `fastmri/data/mri_data.py` (`SliceDataset`,
  `CombinedSliceDataset`, `FastMRIRawDataSample`, metadata extraction), README and
  data documentation. `reconstruction_rss`/`reconstruction_esc` are magnitude
  image targets; raw k-space and volume attributes are separate.
- **Useful implementation:** volume-level sampling and acquisition metadata,
  rather than treating randomly split correlated slices as independent exams.
- **Hardware/data assumptions:** HDF5 cohorts and acquisition/contrast differences;
  downloading requires upstream access/agreement. Archive status and old dependency
  pins make wholesale package adoption inappropriate.
- **License / direct use:** MIT software; dataset terms are distinct.
- **Adaptation:** minimal HDF5 volume adapter for already-authorized image targets,
  explicit acquisition filter and subject/volume manifest; preserve a from-scratch
  control and the frozen MRNet validation cohort.
- **Decision/reason:** prepare an optional integration; do not claim pretraining
  or combine unknown contrasts when the actual external data are unavailable.

## Min-SNR

- **Purpose:** a controlled timestep-loss weighting ablation.
- **Exact source inspected:**
  [`guided_diffusion/gaussian_diffusion.py`, `training_losses`, lines 846–947](https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/5189997d6bbde5a1301bcf31a54d2b3589f7309c/guided_diffusion/gaussian_diffusion.py#L846),
  README and ImageNet configurations.
- **Verified equations:** with SNR = alpha_bar/(1-alpha_bar), epsilon weight is
  min(SNR,gamma)/SNR; v weight is min(SNR,gamma)/(SNR+1); x0 weight is
  min(SNR,gamma). Weight **per-volume** MSE before the batch mean. Gamma 5 is a
  reference configuration, not a proven knee-MRI optimum. Retain unweighted
  denoising error for cross-arm evaluation.
- **Hardware assumptions:** reference uses 2D ImageNet latents and multi-GPU ViTs;
  only the weighting mathematics is needed for the compact 3D system.
- **License / direct use:** no top-level license found via GitHub contents/source;
  do not import/vendor this training repository.
- **Decision/reason:** locally implement and algebraically test the published
  equations, cite the ICCV paper, and compare identical architectures/data/updates.
  Do not copy the reference's four-dimensional broadcasting into 5D MRI tensors.
- **Measured follow-up:** the matched 2,000-update one-MRI-latent v-prediction
  diagnostic was worse with gamma 5 than without weighting (mean reconstruction
  SSIM .5391 versus .6379). This negative tiny-data result does not establish that
  Min-SNR is generally ineffective; it does reject an automatic improvement claim
  for this configuration. Full-cohort adoption remains gated.

## vector-quantize-pytorch

- **Purpose:** controlled quantizer alternatives and healthy-codebook techniques.
- **Source inspected:** `vector_quantize_pytorch/vector_quantize_pytorch.py`
  (`VectorQuantize`, EMA cluster sizes, code expiration, commitment weighting),
  `residual_vq.py` (`ResidualVQ` residual codebooks),
  `finite_scalar_quantization.py` (`FSQ`, scalar bounding/straight-through rounding),
  package metadata and MIT license.
- **Useful implementation:** expired codes are replaced from training samples;
  EMA counts and commitment objective have distinct roles. FSQ removes learned
  codebook optimization; residual stacks change the representation bitrate.
- **Hardware assumptions:** extra residual codebooks add memory and token budget;
  comparisons must account for actual latent dimensions/bitrate.
- **Direct use/adaptation:** MIT imports are possible, but require additional
  einx/einops dependencies. The existing implementation already has EMA/reset.
- **Decision/reason:** first isolate codebook capacity with 256 versus 128 codes.
  Baseline uses 125/128 codes, so collapse is not the leading observed failure.
  ResidualVQ/FSQ/SimVQ remain optional follow-ups if matched KL/VQ evidence points
  specifically to quantization, rather than a blind quantizer sweep.

## 3D MedDiffusion

- **Purpose:** patch-volume encoding and local/global high-resolution lessons.
- **Source inspected:** `AutoEncoder/model/PatchVolume.py` (`patchvolumeAE`,
  `patch_encode`, `patch_encode_sliding`, `decode_sliding`, two-stage forward),
  `AutoEncoder/model/codebook.py`, `config/PatchVolume_4x.yaml`, README and requirements.
- **Useful implementation:** separately patch-wise encode, assemble a full latent
  lattice, volume-wise decode; patch-only decoding introduces context/boundary issues.
- **Hardware assumptions:** configuration has width 48, 8,192 codes, batch 16,
  million-step budget and staged adversarial/perceptual terms. README explicitly
  requests at least 40 GB for all supported inference resolutions; diffusion
  example launches eight GPUs. Requirements contain machine-local Linux wheels.
- **License / direct use:** no clear top-level license; do not vendor code.
- **Decision/reason:** borrow the architectural question from the paper, not the
  implementation/configuration. Full-volume small KL is first; patch training and
  sliding-window behavior are optional, explicitly tested memory fallbacks.

## TCAM-Diff

- **Purpose:** possible later triplane compression alternative.
- **Actual inspected files:** only `README.md`, `environment.yml`, `LICENSE` are
  present at the inspected main SHA. GitHub branch listing has main only and code
  search returns no CrossAttention implementation. `triplanes/`, `diffusion/` and
  training/config paths advertised by README are not present.
- **License:** Apache-2.0 for available material; missing code is not inspectable.
- **Hardware assumptions:** claims cover 128³–512³ non-knee data; no runnable
  source/configuration is available here to validate memory claims.
- **Decision/reason:** research-only; no replacement of the dense KL study and no
  claim that actual triplane classes were inspected or reproduced.

## MedicalNet

- **Purpose:** optional frozen 3D feature diagnostics/perceptual ablation.
- **Source inspected:** `models/resnet.py` (`ResNet`, `resnet10`, dilated stages,
  `conv_seg` head), README, LICENSE; MONAI's `resnet.py` MedicalNet loading path
  and `PerceptualLoss` feature/intensity normalization implementation.
- **Useful implementation:** small 3D ResNet feature extractor with a frozen
  backbone. Original `forward` includes a segmentation head, so its output is
  not automatically a meaningful pooled perceptual embedding.
- **Hardware assumptions:** resnet10 is about 14M parameters, larger than the
  proposed autoencoders; frozen feature losses still retain input gradients and
  consume memory. Original code assumes PyTorch 0.4.1/CUDA 9 and needs adaptation.
- **License / direct use:** LICENSE text explicitly grants MIT (GitHub metadata
  is NOASSERTION). Prefer maintained MONAI architecture plus explicit local weights.
- **Decision/reason:** optional hashed frozen-feature hook; no random encoder
  called "medical pretrained", no MedicalNet distance called a clinical metric,
  and no performance numbers without verified weights/protocol.
- **Completed adaptation:** `scripts/export_medical_features.py` explicitly
  downloads the official public ResNet10 checkpoint at Hugging Face revision
  `deeeaa1e2a342b63e28b61e9d993b63e79c4b437`. Its SHA-256 is
  `afa8055f3e47f4a18239495d92a7abc587902c69c31c743de2b2784653b72605`.
  The maintained MONAI backbone excludes the classification head, permutes the
  input to depth/row/column, resizes to 32×64×64 and applies volume z-scoring.
  The frozen export, its preprocessing and checksum are recorded in
  `runs/MEDICALNET-FEATURES/extractor.json`; data/weights are not redistributed.
- **Measured follow-up:** frozen-feature paired/neighborhood diagnostics completed;
  a matched single-seed perceptual-loss arm did not improve the structural arm's
  PSNR/SSIM or pass the fine-detail gate. Very small cosine distances and
  rank-limited small-sample distribution estimates are descriptive, not evidence
  of knee anatomy or lesion preservation.

## NVIDIA EDM

- **Purpose:** later noise-preconditioning/sampler design reference.
- **Source inspected:** `training/loss.py` (`EDMLoss`, log-normal sigma sampling,
  sigma-dependent weight), `training/networks.py` (`EDMPrecond` c_skip/c_out/c_in
  and log-sigma embedding), `generate.py` sampler and LICENSE.txt.
- **Hardware assumptions:** 2D image networks with continuous-noise training;
  existing discrete-time checkpoints do not become EDM models by swapping a sampler.
- **License / direct use:** CC BY-NC-SA 4.0, not the permissive license of MONAI.
- **Decision/reason:** research-only later ablation. No copied implementation,
  no immediate rewrite, and no claim of evaluating EDM with ordinary DDPM weights.

## Additional spatial-memorization diagnostic: CoordConv

- **Purpose:** test whether explicit index information helps a convolutional
  denoiser intentionally memorize one fixed training lattice at high noise.
- **Source inspected:** `CoordConv.py`, especially `AddCoords`, plus README and
  LICENSE at the revision above. The reference creates normalized 2D index
  channels; the relevant idea is coordinate access, not its TensorFlow API.
- **License / direct use:** source comments and repository-level non-commercial
  terms differ; no source is imported or copied. The local implementation uses
  PyTorch's ordinary `linspace`/`meshgrid` to express the paper's idea in 3D.
- **Adaptation:** optional row/column/slice channels in [-1,1], never patient-space
  coordinates. Shared weights and subsequent RNG state match the non-coordinate
  model; extra input weights start at zero, preserving the initial function.
  This adds parameters and a spatial prior, so it is an architecture ablation.
- **Decision/reason:** run only a matched-update technical memorization test after
  the ordinary and Min-SNR one-volume tests failed. Any improvement here would
  demonstrate training reproduction, not held-out generation or pathology.
- **Measured follow-up:** both 2,000-update coordinate runs completed. Unweighted
  SSIM rises modestly to .6614 (.6379 without coordinates); adding gamma-5 Min-SNR
  yields .5794. Both still fail the fixed SSIM ≥.9 / latent-MSE ≤.1 gate, so no
  four-volume or larger diffusion run was authorized.

## Primary papers and differences from this project

| Idea | Primary citation | Adopted/testable component and important difference |
|---|---|---|
| KL latent diffusion | Rombach, Blattmann, Lorenz, Esser & Ommer, *High-Resolution Image Synthesis with Latent Diffusion Models*, CVPR 2022, [arXiv:2112.10752](https://arxiv.org/abs/2112.10752) | Compact 3D posterior autoencoder; no text/ImageNet training, GAN or pretrained natural-image perceptual default |
| KL inference/training | Kingma & Welling, *Auto-Encoding Variational Bayes*, ICLR 2014, [arXiv:1312.6114](https://arxiv.org/abs/1312.6114) | Reparameterized training, deterministic posterior-mean reconstruction; record reduction and scaling |
| Min-SNR | Hang et al., *Efficient Diffusion Training via Min-SNR Weighting Strategy*, ICCV 2023, pp.7441–7451, [arXiv:2303.09556](https://arxiv.org/abs/2303.09556) | Published weighting equations on small 3D latents, not reference multi-GPU ImageNet experiment |
| MRI augmentation | Pérez-García, Sparks & Ourselin, *TorchIO*, Computer Methods and Programs in Biomedicine 2021, [doi:10.1016/j.cmpb.2021.106236](https://doi.org/10.1016/j.cmpb.2021.106236) | Conservative acquisition-aware transforms; no physical-space claims for MRNet's array grid |
| Dense knee annotations | Desai et al., *SKM-TEA*, NeurIPS Datasets and Benchmarks 2021, [OpenReview](https://openreview.net/forum?id=YDMFgD_qJuA) | Source of future tissue/ROI validation protocols; qDESS differs from MRNet |
| Knee pretraining | Knoll et al., *fastMRI: A Publicly Available Raw k-Space and DICOM Dataset of Knee Images…*, Radiology: AI 2020, [doi:10.1148/ryai.2020190007](https://doi.org/10.1148/ryai.2020190007) | Optional volume-level image-target import; labels/acquisitions are not MRNet equivalents |
| Patch-volume/global-local | Wang et al., *3D MedDiffusion*, 2024 preprint / 2025 revision, [arXiv:2412.13059](https://arxiv.org/abs/2412.13059) | Constrained memory lessons only, not a claimed high-resource reproduction |
| Triplanes | Zhang, Ehinger & Drummond, *TCAM-Diff*, AAAI 2025, [doi:10.1609/aaai.v39i21.34433](https://doi.org/10.1609/aaai.v39i21.34433) | Deferred alternative; source absent in current repository |
| Medical features | Chen, Ma & Zheng, *Med3D*, 2019 preprint, [arXiv:1904.00625](https://arxiv.org/abs/1904.00625) | Optional representation diagnostic, not knee-specific anatomical validation |
| Continuous noise design | Karras, Aittala, Aila & Laine, *Elucidating the Design Space…*, NeurIPS 2022, [arXiv:2206.00364](https://arxiv.org/abs/2206.00364) | Later compatible-training ablation; no immediate sampler transplant |
| Coordinate access | Liu et al., *An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution*, NeurIPS 2018, [arXiv:1807.03247](https://arxiv.org/abs/1807.03247) | Optional 3D array-index channels in a matched one-volume memorization diagnostic; no anatomical coordinates inferred |

Per-component completion and measured results are recorded in
`docs/research_v2_report.md`; this review does not itself claim successful training.
