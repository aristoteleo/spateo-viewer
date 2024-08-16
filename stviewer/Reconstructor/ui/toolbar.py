try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from typing import Optional

from pyvista import BasePlotter
from trame.widgets import html, vuetify

from stviewer.assets import icon_manager
from stviewer.Reconstructor.pv_pipeline import Viewer

from .utils import button, checkbox

# -----------------------------------------------------------------------------
# GUI- UI title
# -----------------------------------------------------------------------------


def ui_title(
    layout, title_name="SPATEO VIEWER", title_icon: Optional[str] = None, **kwargs
):
    """
    Define the title name and logo of the UI.
    Reference: https://trame.readthedocs.io/en/latest/trame.ui.vuetify.html#trame.ui.vuetify.SinglePageWithDrawerLayout

    Args:
        layout: The layout object.
        title_name: Title name of the GUI.
        title_icon: Title icon of the GUI.
        **kwargs: Additional parameters that will be passed to ``html.Img`` function.

    Returns:
        None.
    """

    # Update the toolbar's name
    layout.title.set_text(title_name)
    layout.title.style = (
        "font-family:arial; font-size:25px; font-weight: 550; color: gray;"
    )

    # Update the toolbar's icon
    if not (title_icon is None):
        with layout.icon as icon:
            icon.style = "margin-left: 10px;"  # "width: 7vw; height: 7vh;"
            html.Img(src=title_icon, height=40, **kwargs)


# -----------------------------------------------------------------------------
# GUI- standard ToolBar
# -----------------------------------------------------------------------------


def toolbar_widgets(server, plotter: BasePlotter):
    """
    Generate standard widgets for ToolBar.

    Args:
        server: The trame server.
        plotter: The PyVista plotter to connect with the UI.
    """
    viewer = Viewer(server=server, plotter=plotter)

    vuetify.VSpacer()
    # Upload file
    vuetify.VFileInput(
        v_model=(viewer.UPLOAD_ANNDATA, None),
        label="Select Sample",
        show_size=True,
        small_chips=True,
        truncate_length=25,
        dense=True,
        outlined=True,
        hide_details=True,
        classes="ml-8",
        style="max-width: 300px;",
        rounded=True,
        accept=".h5ad",
        __properties=["accept"],
    )

    vuetify.VSpacer()
    # Change the selection mode
    with vuetify.VBtnToggle(v_model=(viewer.PICKING_MODE, "hover"), dense=True):
        with vuetify.VBtn(value=("item.value",), v_for="item, idx in modes"):
            vuetify.VIcon("{{item.icon}}")
    # Whether to reload the main model
    button(
        click=viewer.on_reload_main_model,
        icon="mdi-restore",
        tooltip="Reload main model",
    )

    vuetify.VProgressLinear(
        indeterminate=True, absolute=True, bottom=True, active=("trame__busy",)
    )


def ui_toolbar(
    server,
    layout,
    plotter: BasePlotter,
    ui_name: str = "SPATEO VIEWER",
    ui_icon=icon_manager.spateo_logo,
):
    """
    Generate standard ToolBar for Spateo UI.

    Args:
        server: The trame server.
        layout: The layout object.
        plotter: The PyVista plotter to connect with the UI.
        ui_name: Title name of the GUI.
        ui_icon: Title icon of the GUI.
    """

    # -----------------------------------------------------------------------------
    # Title
    # -----------------------------------------------------------------------------
    ui_title(layout=layout, title_name=ui_name, title_icon=ui_icon)

    # -----------------------------------------------------------------------------
    # ToolBar
    # -----------------------------------------------------------------------------
    with layout.toolbar as tb:
        tb.height = 55
        tb.dense = True
        tb.clipped_right = True
        toolbar_widgets(server=server, plotter=plotter)
