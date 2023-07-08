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
    ui_standard_drawer,
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

# Generate anndata object
plotter = create_plotter()
init_anndata_path = local_dataset_manager.drosophila_E7_8h_anndata
main_model, active_model, init_scalar, pdd, cdd = init_models(
    plotter=plotter, anndata_path=init_anndata_path
)

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
        # active model
        "activeModel": vtk_mesh(
            active_model,
            point_arrays=[key for key in pdd.keys()],
            cell_arrays=[key for key in cdd.keys()],
        ),
        "activeModelVisible": True,
        # slices alignment
        "slices_alignment": False,
        "slices_key": "slices",
        "slices_align_device": "CPU",
        "slices_align_method": "Paste",
        "slices_align_factor": 0.1,
        "slices_align_max_iter": 200,
        # reconstructed mesh model
        "meshModel": None,
        "meshModelVisible": False,
        "reconstruct_mesh": False,
        "mc_factor": 1.0,
        "mesh_voronoi": 20000,
        "mesh_smooth_factor": 2000,
        "mesh_scale_factor": 1.0,
        "clip_pc_with_mesh": False,
        # output path
        "activeModel_output": None,
        "mesh_output": None,
        "anndata_output": None,
        # Fields available
        "scalar": "area",
        "scalarParameters": {**pdd, **cdd},
        "picking_group": None,
        "overwrite": False,
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
    ui_standard_drawer(layout=layout)

    # -----------------------------------------------------------------------------
    # Main Content
    # -----------------------------------------------------------------------------
    ui_standard_container(server=interactive_server, layout=layout)

    # -----------------------------------------------------------------------------
    # Footer
    # -----------------------------------------------------------------------------
    layout.footer.hide()
    # layout.flush_content()
