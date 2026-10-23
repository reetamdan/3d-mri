"""Publish actual completed stage-one measurements, rejecting mixed cohorts/budgets."""
import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any

if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.integrity import sha256_file
from src.utils.logging import write_json


def summarize(studies, output):
    records, fingerprints, budgets = [], set(), set()
    pending = []
    seen = set()
    for directory in studies:
        root = Path(directory)
        if not (root/'study.json').is_file():
            pending.append(str(root)); continue
        study = json.loads((root/'study.json').read_text())
        configs = study['plan']['configs']
        if any(c.get('smoke_test', False) for c in configs.values()):
            raise ValueError('Artificial smoke runs cannot supply research tables')
        fingerprints.add(study['plan']['dataset_fingerprint'])
        for record in study['records']:
            if (record['arm'], record['seed']) in seen:
                raise ValueError('Duplicate arm/seed; do not silently select the favorable run')
            seen.add((record['arm'], record['seed']))
            config = configs[record['arm']]
            expected = config['epochs']*config['steps_per_epoch']
            history = json.loads((root/record['name']/'training/history.json').read_text())
            actual = sum(r['optimizer_steps'] for r in history)
            if actual != expected:
                raise ValueError('Incomplete optimization budget cannot enter a completed comparison')
            path = root/record['checkpoint']
            if sha256_file(path) != record['checkpoint_sha256']:
                raise ValueError('Measured checkpoint hash changed')
            budgets.add((actual, config['batch_size']*config['gradient_accumulation'], tuple(config['shape'])))
            records.append(dict(record, source_study=str(root), actual_updates=actual))
        if study['status'] != 'completed':
            pending.append(str(root))
    if len(fingerprints) != 1 or len(budgets) != 1:
        raise ValueError('Only the same frozen cohort, resolution and matched update/batch budget can be combined')
    grouped: dict[str, Any] = {}
    for arm in sorted({r['arm'] for r in records}):
        rows = [r for r in records if r['arm'] == arm]
        keys = ('psnr', 'ssim', 'ms_ssim', 'gradient_energy_ratio', 'high_frequency_energy_ratio', 'edge_mae')
        grouped[arm] = dict(seeds=[r['seed'] for r in rows], spatial_compression=rows[0]['compression'],
            metrics={key: dict(mean=statistics.mean(r['metrics'][key] for r in rows),
                               seed_sd=statistics.stdev(r['metrics'][key] for r in rows) if len(rows) > 1 else None) for key in keys},
            parameters=rows[0]['parameters'], train_peak_mib=max(r['training_peak_cuda_mib'] or 0 for r in rows),
            gates_passed=sum(r['quantitative_gate_pass'] for r in rows), anatomy_metric=None)
    def number(item, digits):
        mean = f'{item["mean"]:.{digits}f}'
        return mean+f' ± {item["seed_sd"]:.{digits}f}' if item['seed_sd'] is not None else mean+' (one seed)'
    lines = ['# Measured autoencoder comparisons', '',
        'Same frozen 96/24 model-selection cohort; no independent patient-test claim.',
        'Means ± sample SD describe actual training seeds; single-seed screens have no invented seed variance.', '',
        '| Arm | Seeds | Spatial compression | PSNR dB | SSIM | MS-SSIM | Gradient ratio | HF ratio | Peak train MiB | Gate passes |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---|']
    for arm, row in grouped.items():
        m = row['metrics']
        lines.append(f'| {arm} | '+','.join(map(str, row['seeds']))+' | '+'×'.join(map(str, row['spatial_compression']))
            +f' | {number(m["psnr"], 3)} | {number(m["ssim"], 4)} | {m["ms_ssim"]["mean"]:.4f}'
            +f' | {m["gradient_energy_ratio"]["mean"]:.3f} | {m["high_frequency_energy_ratio"]["mean"]:.3f}'
            +f' | {row["train_peak_mib"]:.1f} | {row["gates_passed"]}/{len(row["seeds"])} |')
    lines += ['', 'Anatomy/lesion proxy values are unavailable: no aligned masks/validated knee model were supplied.',
              'Gradient/frequency metrics use the common resized array grid; they are not anatomical annotations.',
              'Continuous latent scalar counts and discrete-code bitrates differ; spatial factors are not codec compression rates.', '',
              '## Every completed run', '', '| Arm | Seed | PSNR | SSIM | Updates | Effective batch | Decision |',
              '|---|---:|---:|---:|---:|---:|---|']
    batch = next(iter(budgets))[1]
    for row in records:
        lines.append(f'| {row["arm"]} | {row["seed"]} | {row["metrics"]["psnr"]:.5f} | {row["metrics"]["ssim"]:.5f}'
                     +f' | {row["actual_updates"]} | {batch} | '+('quantitative pass; visual review required' if row['quantitative_gate_pass'] else 'detail gate failed')+' |')
    if pending:
        lines += ['', 'Incomplete/pending studies excluded from completed results: '+', '.join(pending)]
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_json(path.with_suffix('.json'), dict(groups=grouped, records=records, dataset_fingerprint=next(iter(fingerprints)),
                                             budget=list(next(iter(budgets))), pending=pending))
    print('Wrote measured table:', path)
    return grouped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--studies', nargs='+', required=True)
    parser.add_argument('--output', default='docs/results/ae_bakeoff_measured.md')
    args = parser.parse_args()
    summarize(args.studies, args.output)


if __name__ == '__main__':
    main()
