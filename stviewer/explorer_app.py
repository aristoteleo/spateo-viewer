try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from tkinter import Tk, filedialog

import matplotlib.pyplot as plt
from trame.widgets import trame as trame_widgets

from .assets import icon_manager, local_dataset_manager
from .Explorer import (
    create_plotter,
    init_actors,
    init_card_parameters,
    init_custom_parameters,
    init_interpolation_parameters,
    init_mesh_parameters,
    init_morphogenesis_parameters,
    init_output_parameters,
    init_pc_parameters,
    ui_container,
    ui_drawer,
    ui_layout,
    ui_toolbar,
)
from .server import get_trame_server

# export WSLINK_MAX_MSG_SIZE=1000000000    # 1GB

# Get a Server to work with
static_server = get_trame_server(name="spateo_explorer")
state, ctrl = static_server.state, static_server.controller
state.trame__title = "SPATEO VIEWER"
state.trame__favicon = icon_manager.spateo_logo
state.setdefault("active_ui", None)

# Generate a new plotter
plotter = create_plotter()
# Init model
(
    anndata_info,
    actors,
    actor_names,
    actor_tree,
    custom_colors,
) = init_actors(
    plotter=plotter,
    path=local_dataset_manager.mouse_E95,
)

# Init parameters
state.update(init_card_parameters)
state.update(init_pc_parameters)
state.update(init_mesh_parameters)
state.update(init_morphogenesis_parameters)
state.update(init_interpolation_parameters)
state.update(init_output_parameters)
state.update(
    {
        "init_dataset": True,
        "anndata_info": anndata_info,
        "pc_obs_value": "mapped_celltype",
        "available_obs": ["None"] + anndata_info["anndata_obs_keys"],
        "pc_gene_value": None,
        "available_genes": ["None"] + anndata_info["anndata_var_index"],
        "pc_colormaps_list": ["spateo_cmap"] + custom_colors + plt.colormaps(),
        # setting
        "actor_ids": actor_names,
        "pipeline": actor_tree,
        "active_id": 1,
        "active_ui": actor_names[0],
        "active_model_type": str(actor_names[0]).split("_")[0],
        "vis_ids": [
            i for i, actor in enumerate(plotter.actors.values()) if actor.visibility
        ],
    }
)
# Custom init parameters
if init_custom_parameters["custom_func"] is True:
    state.update(init_custom_parameters)
else:
    state.update({"custom_func": False})


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
    ui_toolbar(
        server=static_server,
        layout=layout,
        plotter=plotter,
        mode="trame",
        ui_name="SPATEO VIEWER (EXPLORER)",
    )

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
