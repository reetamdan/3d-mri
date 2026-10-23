"""Explicitly fetch a pinned official MedicalNet checkpoint and export a frozen descriptor."""
import argparse
from pathlib import Path
import sys
from urllib.request import urlopen

if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.models.medical_features import load_medicalnet
from src.training.common import fresh_output
from src.utils.integrity import sha256_file
from src.utils.logging import write_json

REVISION = 'deeeaa1e2a342b63e28b61e9d993b63e79c4b437'
WEIGHTS_SHA256 = 'afa8055f3e47f4a18239495d92a7abc587902c69c31c743de2b2784653b72605'
URL = f'https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet10/resolve/{REVISION}/resnet_10_23dataset.pth'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--weights', help='Already downloaded official checkpoint; otherwise fetch the pinned public file')
    args = parser.parse_args()
    out = fresh_output(args.output)
    weights = Path(args.weights) if args.weights else out/'resnet_10_23dataset.pth'
    if args.weights is None:
        partial = weights.with_suffix('.partial')
        count = 0
        with urlopen(URL, timeout=90) as response, partial.open('wb') as file:
            while chunk := response.read(1024**2):
                count += len(chunk)
                if count > 128*1024**2:
                    raise ValueError('MedicalNet checkpoint exceeds expected download budget')
                file.write(chunk)
        partial.replace(weights)
    torch.set_num_threads(4)
    model = load_medicalnet(weights, WEIGHTS_SHA256)
    traced = torch.jit.trace(model, torch.zeros(1, 1, 128, 128, 32), check_trace=False)
    traced.save(str(out/'features.pt'))
    write_json(out/'extractor.json', dict(source_url=URL, revision=REVISION, weight_sha256=WEIGHTS_SHA256,
        feature_sha256=sha256_file(out/'features.pt'), feature_file='features.pt', license='MIT',
        architecture='MONAI 1.6.1 ResNet10, MedicalNet 23-dataset encoder, no classification head',
        input_layout='[B,1,row,column,slice]', preprocess='Permute to depth,row,column; trilinear 32x64x64; volume z-score',
        suitability='Generic 3D medical representation, not a knee-validated anatomy or pathology metric'))
    print('Exported pinned frozen MedicalNet descriptor:', out/'features.pt')


if __name__ == '__main__':
    main()
