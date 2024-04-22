import logging
from kedro.io import AbstractVersionedDataSet
from pathlib import PurePosixPath
from kedro.io.core import (
    get_filepath_str,
    get_protocol_and_path,
    PROTOCOL_DELIMITER,
    Version,
)
import fsspec
import pyarrow.parquet as pq
import pyarrow as pa
from typing import Any, Dict
from copy import deepcopy

logger = logging.getLogger(__name__)


## Inspiration from https://docs.kedro.org/en/0.18.8/_modules/kedro_datasets/pandas/parquet_dataset.html
class CustomParquetDataset(AbstractVersionedDataSet):
    def __init__(
        self,
        filepath: str,
        version: Version = None,
        credentials: Dict[str, Any] = None,
        fs_args: Dict[str, Any] = None,
        compression: str = "ZSTD",
    ) -> None:
        _fs_args = deepcopy(fs_args) or {}
        _credentials = deepcopy(credentials) or {}

        protocol, path = get_protocol_and_path(filepath, version)

        if protocol == "file":
            _fs_args.setdefault("auto_mkdir", True)

        self._protocol = protocol
        self._storage_options = {**_credentials, **_fs_args}
        self._fs = fsspec.filesystem(self._protocol, **self._storage_options)

        super().__init__(
            filepath=PurePosixPath(path),
            version=version,
            exists_function=self._fs.exists,
            glob_function=self._fs.glob,
        )

        # self._filepath = PurePosixPath(path)
        # self._fs = fsspec.filesystem(self._protocol)
        self._compression = compression

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
            protocol=self._protocol,
            compression=self._compression,
            version=self._version,
        )

    def _load(self) -> pa.Table:
        load_path = str(self._get_load_path())
        if self._protocol == "file":
            return pq.read_table(load_path)

        load_path = f"{self._protocol}{PROTOCOL_DELIMITER}{load_path}"
        return pq.read_table(load_path)

    def _save(self, data: pa.Table) -> None:
        save_path = get_filepath_str(self._filepath, self._protocol)
        if self._protocol == "file":
            pq.write_table(data, save_path, compression=self._compression)
        else:
            save_path = f"{self._protocol}{PROTOCOL_DELIMITER}{save_path}"
            pq.write_table(data, save_path, compression=self._compression)
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """Invalidate underlying filesystem caches."""
        filepath = get_filepath_str(self._filepath, self._protocol)
        self._fs.invalidate_cache(filepath)
