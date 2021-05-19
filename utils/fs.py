import os
from typing import Union, Optional, Iterable

AnyFile = Union['File', os.PathLike, str]
AnyFolder = Union['Folder', os.PathLike, str]


class File:

    def __init__(self, path: os.PathLike):
        self._path = os.fspath(path)

    def __fspath__(self):
        return str(self)

    def __str__(self):
        return self._path

    def exists(self) -> bool:
        return os.path.isfile(self)

    def remove(self):
        os.remove(self)

    def unzip(self, to: AnyFolder = '.', specific_members: Optional[Iterable[str]] = None):
        import zipfile

        with zipfile.ZipFile(str(self)) as zf:
            zf.extractall(to, members=specific_members)
