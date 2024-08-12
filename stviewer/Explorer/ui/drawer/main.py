try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from pyvista import BasePlotter
from trame.widgets import vuetify


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
    from stviewer.Explorer.pv_pipeline import PVCB

    PVCB(server=server, plotter=plotter, suppress_rendering=mode == "client")

    with layout.drawer as dr:
        # Pipeline
        from .pipeline import pipeline_panel

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
                items = (
                    items + ["Custom"] if server.state.custom_func is True else items
                )
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
                        from .model_point import pc_card_content

                        pc_card_content()
                    with vuetify.VTabItem(value=(1,)):
                        from .morphogenesis import morphogenesis_card_content

                        morphogenesis_card_content()
                    # Custom
                    if server.state.custom_func is True:
                        with vuetify.VTabItem(value=(2,)):
                            from .custom_card import custom_card_content

                            custom_card_content()
            with vuetify.VCardText(
                classes="py-2",
                v_show=f"active_model_type === 'Mesh'",
                v_if=("show_model_card",),
            ):
                from .model_mesh import mesh_card_content

                mesh_card_content()

        # Output
        from .output import output_panel

        output_panel()
