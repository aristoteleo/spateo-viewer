try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from typing import Optional

from anndata import AnnData
from pyvista import BasePlotter

from .assets import icon_manager
from .server import get_trame_server
from .ui import (
    ui_layout,
    ui_standard_container,
    ui_standard_drawer,
    ui_standard_toolbar,
)
from .dataset import drosophila_E7_9h_dataset
from .pv_pipeline import drosophila_actors


def standard_html(
    plotter: BasePlotter,
    adata: AnnData,
    actors: list,
    actor_names: list,
    tree: Optional[list] = None,
    mode: Literal["trame", "server", "client"] = "trame",
    server_name: Optional[str] = None,
    template_name: str = "main",
    ui_name: str = "SPATEO VIEWER",
    ui_icon=icon_manager.spateo_logo,
    drawer_width: int = 300,
):
    # Get a Server to work with
    server = get_trame_server(name=server_name)
    state, ctrl = server.state, server.controller
    state.trame__title = ui_name
    state.trame__favicon = ui_icon
    state.setdefault("active_ui", None)
    # ctrl.on_server_ready.add(ctrl.view_update)

    # GUI
    ui_standard_layout = ui_layout(
        server=server, template_name=template_name, drawer_width=drawer_width
    )
    with ui_standard_layout as layout:

        # -----------------------------------------------------------------------------
        # ToolBar
        # -----------------------------------------------------------------------------
        ui_standard_toolbar(server=server, layout=layout, plotter=plotter, mode=mode)

        # -----------------------------------------------------------------------------
        # Drawer
        # -----------------------------------------------------------------------------
        ui_standard_drawer(server=server, layout=layout, adata=adata, actors=actors, actor_names=actor_names, tree=tree)

        # -----------------------------------------------------------------------------
        # Main Content
        # -----------------------------------------------------------------------------
        ui_standard_container(server=server, layout=layout, plotter=plotter, mode=mode)

        # -----------------------------------------------------------------------------
        # Footer
        # -----------------------------------------------------------------------------
        layout.footer.hide()

    return server


def stv_html(ui_name: str = "SPATEO VIEWER", **kwargs):
    # Dataset
    (
        adata,
        pc_models,
        pc_model_ids,
        mesh_models,
        mesh_model_ids,
    ) = drosophila_E7_9h_dataset()

    # Pyvista pipeline
    plotter, actors, actor_names, tree = drosophila_actors(
        pc_models=pc_models,
        pc_model_ids=pc_model_ids,
        mesh_models=mesh_models,
        mesh_model_ids=mesh_model_ids,
    )

    # HTML
    server = standard_html(
        plotter=plotter,
        adata=adata,
        actors=actors,
        actor_names=actor_names,
        tree=tree,
        ui_name=ui_name,
        **kwargs
    )
    return server
