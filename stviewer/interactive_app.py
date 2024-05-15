import os

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from trame.widgets import trame as trame_widgets
from vtkmodules.web.utils import mesh as vtk_mesh

from .assets import icon_manager, local_dataset_manager
from .interactive_viewer import (
    create_plotter,
    init_active_parameters,
    init_align_parameters,
    init_custom_parameters,
    init_mesh_parameters,
    init_models,
    init_picking_parameters,
    init_setting_parameters,
    ui_container,
    ui_drawer,
    ui_layout,
    ui_toolbar,
)
from .server import get_trame_server

# export WSLINK_MAX_MSG_SIZE=1000000000    # 1GB

# Get a Server to work with
interactive_server = get_trame_server(name="spateo_interactive_viewer")
state, ctrl = interactive_server.state, interactive_server.controller
state.trame__title = "SPATEO VIEWER"
state.trame__favicon = icon_manager.spateo_logo
state.setdefault("active_ui", None)

# Generate anndata object
plotter = create_plotter()
init_anndata_dir = os.path.join(local_dataset_manager.drosophila_E7_8h, "h5ad")
init_anndata_path = os.path.join(init_anndata_dir, os.listdir(path=init_anndata_dir)[0])
# init_anndata_path = local_dataset_manager.drosophila_E7_8h_anndata
main_model, active_model, init_scalar, pdd, cdd = init_models(
    plotter=plotter, anndata_path=init_anndata_path
)

# Init parameters
state.update(init_active_parameters)
state.update(init_picking_parameters)
state.update(init_align_parameters)
state.update(init_mesh_parameters)
state.update(init_setting_parameters)
state.update(
    {
        "init_anndata": init_anndata_path,
        "upload_anndata": None,
        # main model
        "mainModel": vtk_mesh(
            main_model,
            point_arrays=[key for key in pdd.keys()],
            cell_arrays=[key for key in cdd.keys()],
        ),
        "activeModel": vtk_mesh(
            active_model,
            point_arrays=[key for key in pdd.keys()],
            cell_arrays=[key for key in cdd.keys()],
        ),
        "scalar": "anno_tissue",
        "scalarParameters": {**pdd, **cdd},
    }
)
# Custom init parameters
if init_custom_parameters["custom_func"] is True:
    state.update(init_custom_parameters)
else:
    state.update({"custom_func": False})


# GUI
ui_standard_layout = ui_layout(server=interactive_server, template_name="main")
with ui_standard_layout as layout:
    # Let the server know the browser pixel ratio and the default theme
    trame_widgets.ClientTriggers(
        mounted="pixel_ratio = window.devicePixelRatio, $vuetify.theme.dark = true"
    )

    # -----------------------------------------------------------------------------
    # ToolBar
    # -----------------------------------------------------------------------------
    ui_toolbar(server=interactive_server, layout=layout, plotter=plotter)
    trame_widgets.ClientStateChange(name="activeModel", change=ctrl.view_reset_camera)
    # -----------------------------------------------------------------------------
    # Drawer
    # -----------------------------------------------------------------------------
    ui_drawer(server=interactive_server, layout=layout)

    # -----------------------------------------------------------------------------
    # Main Content
    # -----------------------------------------------------------------------------
    ui_container(server=interactive_server, layout=layout)

    # -----------------------------------------------------------------------------
    # Footer
    # -----------------------------------------------------------------------------
    layout.footer.hide()
    # layout.flush_content()
