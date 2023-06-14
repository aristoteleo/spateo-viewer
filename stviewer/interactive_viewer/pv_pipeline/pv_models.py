import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pyvista as pv

from .pv_plotter import add_single_model


def check_model_data(model, point_data: bool = True, cell_data: bool = True):
    # obtain the data of points
    pdd = {}
    if point_data:
        for name, array in model.point_data.items():
            array = np.asarray(array)
            if len(array.shape) == 1 and name not in [
                "vtkOriginalPointIds",
                "SelectedPoints",
            ]:
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
                }

        # obtain the data of cells
        cdd = {}
        if cell_data:
            for name, array in model.cell_data.items():
                array = np.asarray(array)
                if len(array.shape) == 1 and name not in [
                    "vtkOriginalCellIds",
                    "orig_extract_id",
                ]:
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
                    }

    return model, pdd, cdd


def init_models(plotter, model_path):
    # Generate main model
    main_model = pv.read(model_path)
    main_model.point_data["Default"] = np.ones(shape=(main_model.n_points, 1))
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
