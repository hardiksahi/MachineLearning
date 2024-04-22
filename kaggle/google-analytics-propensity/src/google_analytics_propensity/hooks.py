import logging
import os
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog


class DataCatalogHooks:
    @property
    def _logger(self):
        return logging.getLogger(self.__class__.__name__)

    @hook_impl
    def after_catalog_created(
        self,
        catalog: DataCatalog,
    ) -> None:
        entries = catalog.list()
        for entry in entries:
            try:
                dset = catalog._data_sets.get(entry, None)
                if hasattr(dset, "_path"):
                    self._make_dirs(dset._path)
                elif hasattr(dset, "_filepath"):
                    self._make_dirs(dset._filepath)
                else:
                    pass
            except AttributeError:
                pass

    def _make_dirs(self, path_to_make):
        if not os.path.exists(path_to_make):
            self._logger.info(f"Creating missing path: {path_to_make}")
            os.makedirs(path_to_make)

        # # creates a .gitkeep file while we're at it
        # Path(os.path.join(path_to_make, ".gitkeep")).touch()
