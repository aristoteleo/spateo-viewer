try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from typing import Optional

from anndata import AnnData
from pyvista import BasePlotter
from tkinter import Tk, filedialog
from .assets import icon_manager, local_dataset_manager
from .server import get_trame_server
from .ui import (
    ui_layout,
    ui_standard_container,
    ui_standard_drawer,
    ui_standard_toolbar,
)
from .pv_pipeline import create_plotter, init_actors, all_samples_actors
from .dataset import sample_dataset, abstract_anndata
import anndata as ad
import os

# Get a Server to work with
server =get_trame_server()
state, ctrl = server.state, server.controller
state.trame__title = "SPATEO VIEWER"
state.trame__favicon = icon_manager.spateo_logo
state.setdefault("active_ui", None)

# Generate a new plotter
plotter = create_plotter()
# Init models
state.init_dataset = True
anndata_path, actors, actor_ids, tree = init_actors(plotter=plotter, path=local_dataset_manager.drosophila_E16_17h)
state.actor_ids = actor_ids
state.tree = tree
state.sample_adata_path = anndata_path
state.drawer_content = "drosophila_E16_17h"

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
    server=server, template_name="main", drawer_width=300
)

from trame.widgets import vuetify, html
from reloading import reloading
from trame_server.utils.hot_reload import hot_reload

# Can do `export TRAME_HOT_RELOAD=1` instead
server.hot_reload = True

# Way to enable widgets when UI creation is deferred after server.start()
html.initialize(server)
vuetify.initialize(server)
"""
@ctrl.set("update_drawer")
def update_drawer():
    _update_drawer()

@hot_reload
def _update_drawer():
    #adata = abstract_anndata(path=state.sample_adata_path)
    actors = [value for value in plotter.actors.values()]

    from .ui import pipeline, standard_pc_card, standard_mesh_card
    from .pv_pipeline import PVCB
    with layout.drawer as dr:
        print(state.sample_adata_path)
        pipeline(server=server, actors=actors)
    vuetify.VDivider(classes="mb-2")
    for actor, actor_id in zip(actors, state.actor_ids):
        CBinCard = PVCB(server=server, actor=actor, actor_name=actor_id, adata=adata)
        if str(actor_id).startswith("PC"):
            standard_pc_card(CBinCard, actor_id=actor_id, card_title=actor_id)
        if str(actor_id).startswith("Mesh"):
            standard_mesh_card(CBinCard, actor_id=actor_id, card_title=actor_id)"""

@hot_reload
def _drawer(sample):
    anndata_path = os.path.join(local_dataset_manager[sample], "h5ad")
    adata = abstract_anndata(path=os.path.join(anndata_path, os.listdir(path=anndata_path)[0]))
    sub_plotter, actor_ids, tree = all_samples_actors[sample]
    actors = [value for value in sub_plotter.actors.values()]

    from .ui import pipeline, standard_pc_card, standard_mesh_card
    from .pv_pipeline import PVCB
    pipeline(server=server, actors=actors, actor_ids=actor_ids, actors_tree=tree)
    vuetify.VDivider(classes="mb-2")
    for actor, actor_id in zip(actors, actor_ids):
        CBinCard = PVCB(server=server, actor=actor, actor_name=actor_id, adata=adata)
        if str(actor_id).startswith("PC"):
            standard_pc_card(CBinCard, actor_id=actor_id, card_title=actor_id)
        if str(actor_id).startswith("Mesh"):
            standard_mesh_card(CBinCard, actor_id=actor_id, card_title=actor_id)


with ui_standard_layout as layout:
    # -----------------------------------------------------------------------------
    # ToolBar
    # -----------------------------------------------------------------------------
    ui_standard_toolbar(server=server, layout=layout, plotter=plotter, mode="trame")

    # -----------------------------------------------------------------------------
    # Drawer
    # -----------------------------------------------------------------------------
    with layout.drawer as dr:
        with html.Div(v_show=(f"drawer_content === 'drosophila_E16_17h' ")):
            _drawer("drosophila_E16_17h")
        with html.Div(v_show=f"drawer_content === 'drosophila_E7_9h' "):
            _drawer("drosophila_E7_9h")
        with html.Div(v_show=f"drawer_content === 'drosophila_E9_10h' "):
            _drawer("drosophila_E9_10h")



    #ui_standard_drawer(server=server, layout=layout, plotter=plotter)
    #ctrl.update_drawer()
    # layout.on_server_reload(update_drawer)
    #layout.icon.click = ctrl.update_drawer
    # ctrl.update_drawer()


    # -----------------------------------------------------------------------------
    # Main Content
    # -----------------------------------------------------------------------------
    view = ui_standard_container(server=server, layout=layout, plotter=plotter, mode="trame")

    # -----------------------------------------------------------------------------
    # Footer
    # -----------------------------------------------------------------------------
    layout.footer.hide()
    # layout.flush_content()

