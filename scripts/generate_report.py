from pathlib import Path
import sys
if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import configuration
from src.data.pipeline import inspect_dataset
from src.utils.logging import write_json

def main():
    config = configuration('Generate a dataset quality report')
    _, _, report = inspect_dataset(config)
    output = Path(config['dataset']['metadata_dir']) / 'quality_report.json'
    write_json(output, report)
    print(f'Report saved to {output}')


if __name__ == '__main__':
    main()
