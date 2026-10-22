from pathlib import Path
import sys
if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import configuration
from src.data.pipeline import inspect_dataset
from src.utils.logging import write_csv, write_json

def main():
    config = configuration('Validate indexed MRI files')
    records, errors, report = inspect_dataset(config)
    write_csv(Path(config['logs_dir']) / 'validation_errors.csv', errors, ['scan_id', 'code', 'reason'])
    write_json(Path(config['dataset']['metadata_dir']) / 'validation_report.json', report)
    print(f'Valid: {len(records)}; invalid: {len(errors)}; discovered: {report["discovered_scans"]}')


if __name__ == '__main__':
    main()
