try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from tkinter import Tk, filedialog

from .assets import icon_manager, local_dataset_manager
from .pv_pipeline import create_plotter, init_actors
from .server import get_trame_server
from .ui import (
    ui_layout,
    ui_standard_container,
    ui_standard_drawer,
    ui_standard_toolbar,
)

# export WSLINK_MAX_MSG_SIZE=1000000000    # 1GB

# Get a Server to work with
server = get_trame_server()
state, ctrl = server.state, server.controller
state.trame__title = "SPATEO VIEWER"
state.trame__favicon = icon_manager.spateo_logo
state.setdefault("active_ui", None)

# Generate a new plotter
plotter = create_plotter()
# Init models
state.init_dataset = True
anndata_path, actors, actor_ids, actor_tree, mm_actors, mm_actor_ids = init_actors(
    plotter=plotter,
    path=local_dataset_manager.drosophila_E7_8h,
)
state.sample_adata_path = anndata_path
state.actor_ids = actor_ids
state.pipeline = actor_tree
state.active_id = 0
state.active_ui = actor_ids[0]
state.active_model_type = str(state.active_ui).split("_")[0]
state.mm_actor_ids = mm_actor_ids
state.active_mm_id = None

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
ui_standard_layout = ui_layout(server=server, template_name="main", drawer_width=300)
with ui_standard_layout as layout:
    # -----------------------------------------------------------------------------
    # ToolBar
    # -----------------------------------------------------------------------------
    ui_standard_toolbar(server=server, layout=layout, plotter=plotter, mode="trame")

    # -----------------------------------------------------------------------------
    # Drawer
    # -----------------------------------------------------------------------------
    ui_standard_drawer(server=server, layout=layout, plotter=plotter)

    # -----------------------------------------------------------------------------
    # Main Content
    # -----------------------------------------------------------------------------
    ui_standard_container(server=server, layout=layout, plotter=plotter, mode="trame")

    # -----------------------------------------------------------------------------
    # Footer
    # -----------------------------------------------------------------------------
    layout.footer.hide()
    # layout.flush_content()
