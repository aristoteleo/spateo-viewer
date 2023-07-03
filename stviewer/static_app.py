try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from tkinter import Tk, filedialog

from trame.widgets import trame as trame_widgets

from .assets import icon_manager, local_dataset_manager
from .server import get_trame_server
from .static_viewer import (
    create_plotter,
    init_actors,
    ui_layout,
    ui_standard_container,
    ui_standard_drawer,
    ui_standard_toolbar,
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
anndata_path, actors, actor_ids, actor_tree, mm_actors, mm_actor_ids = init_actors(
    plotter=plotter,
    path=local_dataset_manager.drosophila_E7_8h,
)
# Init parameters
state.update(
    {
        "init_dataset": True,
        "sample_adata_path": anndata_path,
        "actor_ids": actor_ids,
        "pipeline": actor_tree,
        "active_id": 0,
        "active_ui": actor_ids[0],
        "active_model_type": str(state.active_ui).split("_")[0],
        "active_mm_id": None,
        "vis_ids": [
            i for i, actor in enumerate(plotter.actors.values()) if actor.visibility
        ],
        "screenshot_path": None,
        "animation_path": None,
        "animation_npoints": 50,
        "animation_framerate": 10,
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
ui_standard_layout = ui_layout(
    server=static_server, template_name="main", drawer_width=300
)
with ui_standard_layout as layout:
    # Let the server know the browser pixel ratio and the default theme
    trame_widgets.ClientTriggers(
        mounted="pixel_ratio = window.devicePixelRatio, $vuetify.theme.dark = true"
    )

    # -----------------------------------------------------------------------------
    # ToolBar
    # -----------------------------------------------------------------------------
    ui_standard_toolbar(
        server=static_server, layout=layout, plotter=plotter, mode="trame"
    )

    # -----------------------------------------------------------------------------
    # Drawer
    # -----------------------------------------------------------------------------
    ui_standard_drawer(
        server=static_server, layout=layout, plotter=plotter, mode="trame"
    )

    # -----------------------------------------------------------------------------
    # Main Content
    # -----------------------------------------------------------------------------
    ui_standard_container(
        server=static_server, layout=layout, plotter=plotter, mode="trame"
    )

    # -----------------------------------------------------------------------------
    # Footer
    # -----------------------------------------------------------------------------
    layout.footer.hide()
    # layout.flush_content()
