try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from tkinter import Tk, filedialog

import matplotlib.pyplot as plt
from trame.widgets import trame as trame_widgets

from .assets import icon_manager, local_dataset_manager
from .server import get_trame_server
from .static_viewer import (
    create_plotter,
    init_actors,
    init_mesh_parameters,
    init_morphogenesis_parameters,
    init_output_parameters,
    init_pc_parameters,
    ui_container,
    ui_drawer,
    ui_layout,
    ui_toolbar,
)

# export WSLINK_MAX_MSG_SIZE=1000000000    # 1GB

# Get a Server to work with
static_server = get_trame_server(name="spateo_static_viewer")
state, ctrl = static_server.state, static_server.controller
state.trame__title = "SPATEO VIEWER"
state.trame__favicon = icon_manager.spateo_logo
state.setdefault("active_ui", None)

# Generate a new plotter
plotter = create_plotter()
# Init model
(
    anndata_path,
    anndata_metrices,
    actors,
    actor_names,
    actor_tree,
    custom_colors,
) = init_actors(
    plotter=plotter,
    path=local_dataset_manager.drosophila_E7_8h,
)

# Init parameters
state.update(init_pc_parameters)
state.update(init_mesh_parameters)
state.update(init_morphogenesis_parameters)
state.update(init_output_parameters)

state.update(
    {
        "init_dataset": True,
        "anndata_path": anndata_path,
        "matrices_list": anndata_metrices,
        # setting
        "actor_ids": actor_names,
        "pipeline": actor_tree,
        "active_id": 1,
        "active_ui": actor_names[0],
        "active_model_type": str(actor_names[0]).split("_")[0],
        "vis_ids": [
            i for i, actor in enumerate(plotter.actors.values()) if actor.visibility
        ],
        "show_model_card": True,
        "show_output_card": True,
        "pc_colormaps": ["default_cmap"] + custom_colors + plt.colormaps(),
    }
)


# Upload directory
def open_directory():
    dirpath = filedialog.askdirectory(title="Select Directory")
    if not dirpath:
        return
    state.selected_dir = dirpath
    ctrl.view_update()


root = Tk()
root.withdraw()
root.wm_attributes("-topmost", 1)
state.selected_dir = "None"
ctrl.open_directory = open_directory


# GUI
ui_standard_layout = ui_layout(server=static_server, template_name="main")
with ui_standard_layout as layout:
    # Let the server know the browser pixel ratio and the default theme
    trame_widgets.ClientTriggers(
        mounted="pixel_ratio = window.devicePixelRatio, $vuetify.theme.dark = true"
    )

    # -----------------------------------------------------------------------------
    # ToolBar
    # -----------------------------------------------------------------------------
    ui_toolbar(server=static_server, layout=layout, plotter=plotter, mode="trame")

    # -----------------------------------------------------------------------------
    # Drawer
    # -----------------------------------------------------------------------------
    ui_drawer(server=static_server, layout=layout, plotter=plotter, mode="trame")

    # -----------------------------------------------------------------------------
    # Main Content
    # -----------------------------------------------------------------------------
    ui_container(server=static_server, layout=layout, plotter=plotter, mode="trame")

    # -----------------------------------------------------------------------------
    # Footer
    # -----------------------------------------------------------------------------
    layout.footer.hide()
    # layout.flush_content()
