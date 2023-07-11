import warnings

warnings.filterwarnings("ignore")

import anndata as ad
import numpy as np
import pyvista as pv

from .pv_plotter import add_single_model


def check_model_data(model, point_data: bool = True, cell_data: bool = True):
    # obtain the data of points
    pdd = {}
    if point_data:
        for name, array in model.point_data.items():
            if name != "obs_index":
                array = np.asarray(array)
                if len(array.shape) == 1 and name not in [
                    "vtkOriginalPointIds",
                    "SelectedPoints",
                    "vtkInsidedness",
                ]:
                    od = {"None": "None"}
                    if not np.issubdtype(array.dtype, np.number):
                        od = {o: i for i, o in enumerate(np.unique(array).tolist())}
                        model.point_data[name] = np.asarray(
                            list(map(lambda x: od[x], array)), dtype=float
                        )
                        array = np.asarray(model.point_data[name])
                    pdd[name] = {
                        "name": name,
                        "range": [array.min(), array.max()],
                        "value": name,
                        "text": name,
                        "scalarMode": 3,
                        "raw_labels": od,
                    }

    # obtain the data of cells
    cdd = {}
    if cell_data:
        for name, array in model.cell_data.items():
            if name != "obs_index":
                array = np.asarray(array)
                if len(array.shape) == 1 and name not in [
                    "vtkOriginalCellIds",
                    "orig_extract_id",
                    "vtkInsidedness",
                ]:
                    od = {"None": "None"}
                    if not np.issubdtype(array.dtype, np.number):
                        od = {o: i for i, o in enumerate(np.unique(array).tolist())}
                        model.cell_data[name] = np.asarray(
                            list(map(lambda x: od[x], array)), dtype=float
                        )
                        array = np.asarray(model.cell_data[name])
                    cdd[name] = {
                        "name": name,
                        "range": [array.min(), array.max()],
                        "value": name,
                        "text": name,
                        "scalarMode": 3,
                        "raw_labels": od,
                    }

    return model, pdd, cdd


def init_models(plotter, anndata_path):
    # Generate init anndata object
    init_adata = ad.read_h5ad(anndata_path)
    init_adata.obs["Default"] = np.ones(shape=(init_adata.shape[0], 1))
    init_adata.obsm["spatial"] = (
        np.c_[init_adata.obsm["spatial"], np.ones(shape=(init_adata.shape[0], 1))]
        if init_adata.obsm["spatial"].shape[1] == 2
        else init_adata.obsm["spatial"]
    )
    spatial_center = init_adata.obsm["spatial"].mean(axis=0)
    if tuple(spatial_center) != (0, 0, 0):
        init_adata.obsm["spatial"] = init_adata.obsm["spatial"] - spatial_center
    for key in init_adata.obs_keys():
        if init_adata.obs[key].dtype == "category":
            init_adata.obs[key] = np.asarray(init_adata.obs[key], dtype=str)
        if np.issubdtype(init_adata.obs[key].dtype, np.number):
            init_adata.obs[key] = np.asarray(init_adata.obs[key], dtype=float)

    # Construct init pc model
    from .pv_tdr import construct_pc

    main_model = construct_pc(adata=init_adata, spatial_key="spatial")
    _obs_index = main_model.point_data["obs_index"]
    for key in init_adata.obs_keys():
        main_model.point_data[key] = init_adata.obs.loc[_obs_index, key]

    main_model, pdd, cdd = check_model_data(
        model=main_model, point_data=True, cell_data=True
    )
    _ = add_single_model(plotter=plotter, model=main_model, model_name="mainModel")

    # Generate active model
    active_model = main_model.copy()
    _ = add_single_model(
        plotter=plotter,
        model=active_model,
        model_name="activeModel",
    )

    # Init parameters
    init_scalar = "Default"
    return main_model, active_model, init_scalar, pdd, cdd
