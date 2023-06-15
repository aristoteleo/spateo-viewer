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
    init_models,
    ui_layout,
    ui_standard_container,
    ui_standard_toolbar,
)
from .server import get_trame_server

# export WSLINK_MAX_MSG_SIZE=1000000000    # 1GB

# Get a Server to work with
interactive_server = get_trame_server(name="spateo_interactive_viewer")
state, ctrl = interactive_server.state, interactive_server.controller
state.trame__title = "SPATEO VIEWER"
state.trame__favicon = icon_manager.spateo_logo
state.setdefault("active_ui", None)

# Generate inite models
plotter = create_plotter()
model_path = os.path.join(
    local_dataset_manager.drosophila_E7_8h,
    "pc_models/0_Embryo_E7_8h_aligned_pc_model.vtk",
)
main_model, active_model, init_scalar, pdd, cdd = init_models(
    plotter=plotter, model_path=model_path
)

state.update(
    {
        "upload_file_path": None,
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
        # Fields available
        "scalar": init_scalar,
        "scalarParameters": {**pdd, **cdd},
        # picking controls
        "modes": [
            {"value": "hover", "icon": "mdi-magnify"},
            {"value": "click", "icon": "mdi-cursor-default-click-outline"},
            {"value": "select", "icon": "mdi-select-drag"},
        ],
        # Picking feedback
        "pickData": None,
        "selectData": None,
        "resetModel": False,
        "tooltip": "",
        "coneVisibility": False,
        # Main model
        "activeModelVisible": True,
        # Render
        "background_color": "[0, 0, 0]",
        "pixel_ratio": 5,
    }
)

# GUI
ui_standard_layout = ui_layout(
    server=interactive_server, template_name="main", drawer_width=300
)
with ui_standard_layout as layout:
    # Let the server know the browser pixel ratio and the default theme
    trame_widgets.ClientTriggers(
        mounted="pixel_ratio = window.devicePixelRatio, $vuetify.theme.dark = true"
    )

    # -----------------------------------------------------------------------------
    # ToolBar
    # -----------------------------------------------------------------------------
    ui_standard_toolbar(server=interactive_server, layout=layout, plotter=plotter)
    trame_widgets.ClientStateChange(name="activeModel", change=ctrl.view_reset_camera)
    # -----------------------------------------------------------------------------
    # Drawer
    # -----------------------------------------------------------------------------

    # -----------------------------------------------------------------------------
    # Main Content
    # -----------------------------------------------------------------------------
    ui_standard_container(server=interactive_server, layout=layout)

    # -----------------------------------------------------------------------------
    # Footer
    # -----------------------------------------------------------------------------
    layout.footer.hide()
    # layout.flush_content()
