import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pyvista as pv

from .pv_plotter import add_single_model


def init_models(plotter, model_path):
    # Generate main model
    main_model = pv.read(model_path)
    main_model.point_data["Default"] = np.ones(shape=(main_model.n_points, 1))
    _ = add_single_model(
        plotter=plotter, model=main_model, model_style="points", model_name="mainModel"
    )

    # Generate active model
    active_model = main_model.copy()
    _ = add_single_model(
        plotter=plotter,
        model=active_model,
        model_style="points",
        model_name="activeModel",
    )

    # Init parameters
    _scalar = "Default"
    _scalarParameters = {
        name: {
            "name": name,
            "range": [np.asarray(array).min(), np.asarray(array).max()],
            "value": name,
            "text": name,
            "scalarMode": 3,
        }
        for name, array in main_model.point_data.items()
        if len(array.shape) == 1 and array.dtype == "float"
    }

    return main_model, active_model, _scalar, _scalarParameters
