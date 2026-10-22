"""Explicit, size-limited download of three public 3D Slicer MRI samples.

Run manually; never called by app startup. Requires requirements-samples.txt.
Source registration: https://github.com/Slicer/Slicer/blob/main/Modules/Scripted/SampleData/SampleData.py
The two RegLib_C01 scans are timepoints of the same subject.
"""
from pathlib import Path
import hashlib
import json
import urllib.request
import numpy as np
import nibabel as nib
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = [
    ('MRHead', 'slicer_mrhead', 'cc211f0dfd9a05ca3841ce1141b292898b2dd2d3f08286affadf823a7e58df93'),
    ('MRBrainTumor1', 'slicer_reglib_c01', '998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95'),
    ('MRBrainTumor2', 'slicer_reglib_c01', '1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97'),
]
BASE = 'https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/'
MAX_TOTAL = 40 * 1024 * 1024


def download_samples():
    source_dir = ROOT / 'data' / 'downloads' / 'slicer'
    source_dir.mkdir(parents=True, exist_ok=True)
    provenance = []
    total = 0
    for name, subject, digest in SAMPLES:
        source = source_dir / f'{name}.nrrd'
        if not source.exists():
            partial = source.with_suffix('.part')
            with urllib.request.urlopen(BASE + digest, timeout=60) as response, partial.open('wb') as out:
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_TOTAL:
                        raise ValueError('Sample download exceeds the 40 MiB total limit')
                    out.write(chunk)
            partial.replace(source)
        else:
            total += source.stat().st_size
            if total > MAX_TOTAL:
                raise ValueError('Cached sample data exceeds the 40 MiB limit')
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Checksum mismatch: {name}')
        image = sitk.ReadImage(str(source))
        if image.GetDimension() != 3:
            raise ValueError('Expected a scalar 3D source image')
        # SimpleITK handles the LPS-to-RAS coordinate conversion when writing NIfTI.
        output = ROOT / 'data' / 'raw' / subject / f'{name}.nii.gz'
        output.parent.mkdir(parents=True, exist_ok=True)
        if not output.exists():
            # Do not propagate descriptive patient/header fields into the conversion.
            clean = sitk.GetImageFromArray(sitk.GetArrayFromImage(image))
            clean.CopyInformation(image)
            sitk.WriteImage(clean, str(output), True)
        nii = nib.load(output)
        if not isinstance(nii, nib.Nifti1Image):
            raise ValueError('Expected the converted NIfTI-1 sample')
        original = sitk.GetArrayFromImage(image).transpose(2, 1, 0)
        np.testing.assert_array_equal(np.asarray(nii.dataobj), original)
        lps_to_ras = np.diag([-1., -1., 1.])
        expected = np.eye(4)
        expected[:3, :3] = lps_to_ras @ np.array(image.GetDirection()).reshape(3, 3) @ np.diag(image.GetSpacing())
        expected[:3, 3] = lps_to_ras @ image.GetOrigin()
        np.testing.assert_allclose(nii.affine, expected, atol=1e-4)
        provenance.append(dict(sample=name, subject_group=subject, url=BASE+digest,
            source_sha256=digest, source_bytes=source.stat().st_size,
            nifti_path=str(output.relative_to(ROOT)), nifti_bytes=output.stat().st_size,
            nifti_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
            shape=list(nii.shape), spacing=[float(v) for v in nii.header.get_zooms()],
            spatial_unit=nii.header.get_xyzt_units()[0],
            conversion='SimpleITK NRRD to NIfTI; intensities and physical geometry verified unchanged'))
        print(f'{name}: {source.stat().st_size / 1024**2:.2f} MiB downloaded, shape {nii.shape}', flush=True)
    target = ROOT / 'data' / 'metadata' / 'sample_sources.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'source_collection': '3D Slicer public sample data',
        'documentation': 'https://www.slicer.org/wiki/SampleData',
        'purpose': 'Real MRI software smoke testing; not a training or evaluation cohort',
        'download_bytes': total, 'samples': provenance}, indent=2), encoding='utf-8')
    print(f'Total source download: {total / 1024**2:.2f} MiB')


if __name__ == '__main__':
    download_samples()
