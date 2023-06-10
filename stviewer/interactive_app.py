import os

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import pyvista as pv
from tkinter import Tk, filedialog
from trame.widgets import trame

from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import html, trame
from trame.widgets import vtk as vtk_widgets
from trame.widgets import vuetify
from vtkmodules.vtkFiltersCore import vtkThreshold
from vtkmodules.vtkFiltersGeneral import vtkExtractSelectedFrustum
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
from vtkmodules.web.utils import mesh as vtk_mesh

from vtkmodules.vtkFiltersCore import vtkThreshold
from vtkmodules.vtkFiltersGeneral import vtkExtractSelectedFrustum
from .assets import icon_manager, local_dataset_manager, abstract_anndata
from .server import get_trame_server
from .interactive_viewer import (
    create_plotter,
    add_single_model,
    ui_layout,
    ui_standard_container,
    ui_standard_toolbar,
)

# export WSLINK_MAX_MSG_SIZE=1000000000    # 1GB

# Get a Server to work with
interactive_server = get_trame_server(name="spateo_interactive_viewer")
state, ctrl = interactive_server.state, interactive_server.controller
state.trame__title = "SPATEO VIEWER"
state.trame__favicon = icon_manager.spateo_logo
state.setdefault("active_ui", None)

# Generate inite model
init_model_path = os.path.join(local_dataset_manager.drosophila_E7_8h, "pc_models/0_Embryo_E7_8h_aligned_pc_model.vtk")
init_anndata_path = os.path.join(local_dataset_manager.drosophila_E7_8h, "h5ad/E7_8h_cellbin_v3.h5ad")

plotter = create_plotter()
main_model = pv.read(filename=init_model_path)
main_actor = add_single_model(plotter=plotter, model=main_model, model_style="points", model_size=8)

# Init parameters
state.update(
    {
        "init_dataset": True,
        "sample_adata_path": init_anndata_path,
        "MM": None,
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
        "tooltip": "",
        "coneVisibility": False,
        "pixel_ratio": 2,
        # Main model
        "MMVisible": True,
    }
)

# Frustrum extraction
extract = vtkExtractSelectedFrustum()
extract.SetInputData(main_model)

threshold = vtkThreshold()
threshold.SetInputConnection(extract.GetOutputPort())
threshold.SetLowerThreshold(0)
threshold.SetInputArrayToProcess(0, 0, 0, 1, "vtkInsidedness")


# GUI
ui_standard_layout = ui_layout(server=interactive_server, template_name="main", drawer_width=300)
with ui_standard_layout as layout:
    # Let the server know the browser pixel ratio
    trame.ClientTriggers(mounted="pixel_ratio = window.devicePixelRatio")

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
    ui_standard_container(layout=layout, plotter=plotter)

    # -----------------------------------------------------------------------------
    # Footer
    # -----------------------------------------------------------------------------
    layout.footer.hide()
    # layout.flush_content()
