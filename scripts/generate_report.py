from pathlib import Path
from _common import configuration
from src.data.pipeline import inspect_dataset
from src.utils.logging import write_json

if __name__ == '__main__':
    config = configuration('Generate a dataset quality report')
    _, _, report = inspect_dataset(config)
    output = Path(config['dataset']['metadata_dir']) / 'quality_report.json'
    write_json(output, report)
    print(f'Report saved to {output}')
