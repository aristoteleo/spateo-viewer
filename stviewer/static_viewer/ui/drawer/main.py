try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from typing import Optional

import matplotlib.pyplot as plt
from pyvista import BasePlotter
from trame.widgets import trame, vuetify

from stviewer.static_viewer.pv_pipeline import PVCB

from .model_mesh import mesh_card_content
from .model_point import pc_card_content
from .morphogenesis import morphogenesis_card_content
from .output import output_panel
from .pipeline import pipeline_panel


def _get_default_cmap():
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap

    if "default_cmap" not in mpl.colormaps():
        colors = ["#4B0082", "#800080", "#F97306", "#FFA500", "#FFD700", "#FFFFCB"]
        nodes = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        mpl.colormaps.register(
            LinearSegmentedColormap.from_list("default_cmap", list(zip(nodes, colors)))
        )
    return "default_cmap"


def ui_drawer(
    server,
    layout,
    plotter: BasePlotter,
    mode: Literal["trame", "server", "client"] = "trame",
):
    """
    Generate standard Drawer for Spateo UI.

    Args:
        server: The trame server.
        layout: The layout object.
        plotter: The PyVista plotter to connect with the UI.
        mode: The UI view mode. Options are:

            * ``'trame'``: Uses a view that can switch between client and server rendering modes.
            * ``'server'``: Uses a view that is purely server rendering.
            * ``'client'``: Uses a view that is purely client rendering (generally safe without a virtual frame buffer)
    """

    _get_default_cmap()
    PVCB(server=server, plotter=plotter, suppress_rendering=mode == "client")
    with layout.drawer as dr:
        # Pipeline
        pipeline_panel(server=server, plotter=plotter)

        # Active model
        with vuetify.VToolbar(
            dense=True, outlined=True, classes="pa-0 ma-0", style="flex: none;"
        ):
            vuetify.VIcon("mdi-format-paint")
            vuetify.VCardTitle(
                " Active Model",
                classes="pa-0 ma-0",
                style="flex: none;",
                hide_details=True,
                dense=True,
            )

            vuetify.VSpacer()
            with vuetify.VBtn(
                small=True,
                icon=True,
                click="show_model_card = !show_model_card",
            ):
                vuetify.VIcon("mdi-unfold-less-horizontal", v_if=("show_model_card",))
                vuetify.VIcon("mdi-unfold-more-horizontal", v_if=("!show_model_card",))
        with vuetify.VCard(style="flex: none;", classes="pa-0 ma-0"):
            with vuetify.VCardText(
                classes="py-2 ",
                v_show=f"active_model_type === 'PC'",
                v_if=("show_model_card",),
            ):
                items = ["Model", "Morphogenesis"]
                with vuetify.VTabs(v_model=("pc_active_tab", 0), left=True):
                    for item in items:
                        vuetify.VTab(
                            item,
                            style="width: 50%;",
                        )
                with vuetify.VTabsItems(
                    value=("pc_active_tab",),
                    style="width: 100%; height: 100%;",
                ):
                    with vuetify.VTabItem(value=(0,)):
                        pc_card_content()
                    with vuetify.VTabItem(value=(1,)):
                        morphogenesis_card_content()
            with vuetify.VCardText(
                classes="py-2",
                v_show=f"active_model_type === 'Mesh'",
                v_if=("show_model_card",),
            ):
                mesh_card_content()

        # Output
        output_panel()
