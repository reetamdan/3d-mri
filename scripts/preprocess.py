from _common import configuration
from src.data.pipeline import run_pipeline

if __name__ == '__main__':
    result = run_pipeline(configuration('Preprocess and export MRI volumes'))
    print(f'Exported {result["report"]["valid_scans"]} scans to {result["output_dir"]}')
    print(f'Invalid scans: {len(result["errors"])}')
