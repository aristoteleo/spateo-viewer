import base64
import os
from pathlib import Path

from trame.app.mimetypes import to_mime


def to_base64(file_path):
    """
    Return the base64 content of the file path.

    Args:
        file_path: Path to the file to read.

    Return:
        File content encoded in base64

    """
    with open(file_path, "rb") as bin_file:
        return base64.b64encode(bin_file.read()).decode("ascii")


def to_url(file_path: str):
    """
    Return the base64 encoded URL of the file path.

    Args:
        file_path: Path to the file to read.

    Return:
        Inlined bas64 encoded url (data:{mime};base64,{content})
    """
    encoded = to_base64(file_path)
    mime = to_mime(file_path)
    return f"data:{mime};base64,{encoded}"


class LocalFileManager:
    """LocalFileManager provide convenient methods for handling local files"""

    def __init__(self, base_path):
        """
        Provide the base path on which relative path should be based on.
        base_path: A file or directory path
        """
        _base = Path(base_path)

        # Ensure directory
        if _base.is_file():
            _base = _base.parent

        self._root = Path(str(_base.resolve().absolute()))
        self._assests = {}

    def __getitem__(self, name):
        return self._assests.get(name)

    def __getattr__(self, name):
        return self._assests.get(name)

    def _to_path(self, file_path):
        _input_file = Path(file_path)
        if _input_file.is_absolute():
            return str(_input_file.resolve().absolute())

        return str(self._root.joinpath(file_path).resolve().absolute())

    def file_url(self, key: str, file_path: str):
        """
        Store an url encoded file content under the provided key name.

        Args:
            key: The name for that content which can then be accessed by the [] or . notation
            file_path: A file path
        """
        # if file_path is None:
        #    file_path, key = key, file_path

        # data = to_url(self._to_path(file_path))

        if key is not None:
            self._assests[key] = file_path

        return file_path

    def dir_url(self, key: str, dir_path: str):
        """
        Store a directory url under the provided key name.

        Args:
            key: The name for that content which can then be accessed by the [] or . notation
            dir_path: A directory path
        """
        if dir_path is None:
            dir_path, key = key, dir_path

        data = self._to_path(dir_path)

        if key is not None:
            self._assests[key] = data

        return data

    @property
    def assets(self):
        """Return the full set of assets as a dict"""
        return self._assests

    def get_assets(self, *keys):
        """Return a filtered out dict using the provided set of keys"""
        if len(keys) == 0:
            return self.assets

        _assets = {}
        for key in keys:
            _assets[key] = self._assests.get(key)

        return _assets


local_dataset_manager = LocalFileManager(__file__)
if os.path.exists(r"./stviewer/assets/dataset/mouse_E95"):
    local_dataset_manager.dir_url("mouse_E95", r"./dataset/mouse_E95")
if os.path.exists(r"./stviewer/dataset/mouse_E115"):
    local_dataset_manager.dir_url("mouse_E115", r"./dataset/mouse_E115")
if os.path.exists(r"./stviewer/dataset/drosophila_S11"):
    local_dataset_manager.dir_url("drosophila_S11", r"./dataset/drosophila_S11")

if os.path.exists(r"./stviewer/assets/dataset/drosophila_S11/h5ad/S11_cellbin.h5ad"):
    local_dataset_manager.file_url(
        "drosophila_S11_anndata",
        r"./stviewer/assets/dataset/drosophila_S11/h5ad/S11_cellbin.h5ad",
    )

"""
# -----------------------------------------------------------------------------
# Data file information
# -----------------------------------------------------------------------------
from trame.assets.remote import HttpFile
dataset_file = HttpFile(
    "./data/disk_out_ref.vtu",
    "https://github.com/Kitware/trame/raw/master/examples/data/disk_out_ref.vtu",
    __file__,
)
"""
