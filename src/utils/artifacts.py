"""Owner-relative artifact references and relocation of legacy project-local paths."""
import os
from pathlib import Path, PureWindowsPath
import json
from typing import Any


def reference(target: str | Path, owner: str | Path) -> str:
    """Store a reference relative to the directory containing its manifest."""
    try:
        return Path(os.path.relpath(Path(target).resolve(), Path(owner).resolve())).as_posix()
    except ValueError:  # Separate Windows drives cannot form a relative path.
        return str(Path(target).resolve())


def resolve_reference(owner: str | Path, value: str | Path) -> Path:
    owner_path = Path(owner).resolve()
    text = str(value)
    windows_absolute = PureWindowsPath(text).is_absolute()
    path = Path(text) if Path(text).is_absolute() or windows_absolute else owner_path / text
    if path.exists():
        return path.resolve()
    # Old reports embedded a developer's absolute project root. Relocate only
    # recognized project-local artifact trees, never arbitrary external paths.
    if Path(text).is_absolute() or windows_absolute:
        parts = text.replace('\\', '/').split('/')
        for root in (owner_path, *owner_path.parents):
            if (root / 'configs').is_dir() and (root / 'src').is_dir():
                for marker in ('data', 'runs'):
                    if marker in parts:
                        candidate = root.joinpath(*parts[parts.index(marker):])
                        if candidate.exists():
                            return candidate.resolve()
                break
    return path.resolve()


def relocate_project(root: str | Path) -> dict[str, Any]:
    """Rewrite runtime JSON path references, never model/tensor/source bytes."""
    root = Path(root).resolve()
    fields = {'input_dataset', 'data_root', 'vq_checkpoint', 'diffusion_checkpoint', 'source_root', 'output_dir', 'vq_path', 'root'}
    changed = []
    def visit(value: Any, owner: Path) -> Any:
        if isinstance(value, list):
            return [visit(item, owner) for item in value]
        if not isinstance(value, dict):
            return value
        result = {}
        for key, item in value.items():
            if key in fields and isinstance(item, str):
                target = resolve_reference(owner, item)
                result[key] = reference(target, owner) if target.exists() and target.is_relative_to(root) else item
            else:
                result[key] = visit(item, owner)
        return result
    for tree in ('runs', 'data/metadata', 'data/processed'):
        for path in (root/tree).rglob('*.json'):
            original = json.loads(path.read_text(encoding='utf-8-sig'))
            updated = visit(original, path.parent)
            if updated != original:
                partial = path.with_suffix('.relocation-partial')
                partial.write_text(json.dumps(updated, indent=2), encoding='utf-8')
                partial.replace(path)
                changed.append(path.relative_to(root).as_posix())
    return dict(changed=changed, count=len(changed), model_and_tensor_bytes_preserved=True)
