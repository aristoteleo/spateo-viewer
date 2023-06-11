import os

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from tkinter import Tk, filedialog

import pyvista as pv
from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import html
from trame.widgets import trame as trame_widgets
from trame.widgets import vtk as vtk_widgets
from trame.widgets import vuetify
from vtkmodules.vtkFiltersCore import vtkThreshold
from vtkmodules.vtkFiltersGeneral import vtkExtractSelectedFrustum
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
from vtkmodules.web.utils import mesh as vtk_mesh

from .assets import abstract_anndata, icon_manager, local_dataset_manager
from .interactive_viewer import (
    add_single_model,
    create_plotter,
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

# Generate inite model
init_model_path = os.path.join(
    local_dataset_manager.drosophila_E7_8h,
    "pc_models/0_Embryo_E7_8h_aligned_pc_model.vtk",
)
init_anndata_path = os.path.join(
    local_dataset_manager.drosophila_E7_8h, "h5ad/E7_8h_cellbin_v3.h5ad"
)

plotter = create_plotter()
main_model = pv.read(filename=init_model_path)
active_model = main_model.copy()
_ = add_single_model(
    plotter=plotter, model=main_model, model_style="points", model_name="mainModel"
)
_ = add_single_model(
    plotter=plotter, model=active_model, model_style="points", model_name="activeModel"
)

# Init parameters
state.update(
    {
        "init_dataset": True,
        "sample_adata_path": init_anndata_path,
        "mainModel": None,
        "activeModel": None,
        # Fields available
        # "fieldParameters": fieldParameters,
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

    # -----------------------------------------------------------------------------
    # Drawer
    # -----------------------------------------------------------------------------

    # -----------------------------------------------------------------------------
    # Main Content
    # -----------------------------------------------------------------------------
    ui_standard_container(server=interactive_server, layout=layout, plotter=plotter)

    # -----------------------------------------------------------------------------
    # Footer
    # -----------------------------------------------------------------------------
    layout.footer.hide()
    # layout.flush_content()
