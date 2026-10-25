from pathlib import Path
from mri3d.configuration import workspace_root
from app.presentation import overview_page

overview_page(Path(workspace_root()))
