"""Bundle only public presets; configs/ remains their single source of truth."""
from pathlib import Path
import ast
from setuptools.command.build_py import build_py

# PEP 517 loads this file before application packages are importable. Read the
# literal public manifest without importing mri3d or any runtime dependency.
_manifest = ast.parse((Path(__file__).parent/'presets/__init__.py').read_text(encoding='utf-8'))
PRESETS = ast.literal_eval(next(node.value for node in _manifest.body if isinstance(node, ast.Assign)
                               and any(isinstance(target, ast.Name) and target.id == 'PRESETS' for target in node.targets)))


class BuildPy(build_py):
    def run(self):
        super().run()
        source = Path(__file__).resolve().parents[1]/'configs'
        target = Path(self.build_lib)/'mri3d/presets'
        self.mkpath(str(target))
        for name in PRESETS:
            self.copy_file(str(source/name), str(target/name))

    def get_outputs(self, include_bytecode=1):
        return super().get_outputs(include_bytecode) + [str(Path(self.build_lib)/'mri3d/presets'/name)
                                                       for name in PRESETS]
