import sys
from pathlib import Path
if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import configuration
from src.data.pipeline import run_pipeline

def main():
    result = run_pipeline(configuration('Preprocess and export MRI volumes'))
    print(f'Exported {result["report"]["valid_scans"]} scans to {result["output_dir"]}')
    print(f'Invalid scans: {len(result["errors"])}')


if __name__ == '__main__':
    main()
