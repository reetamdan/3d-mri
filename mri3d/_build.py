"""Bundle only public presets; configs/ remains their single source of truth."""
from pathlib import Path
from setuptools.command.build_py import build_py

from mri3d.presets import PRESETS


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
