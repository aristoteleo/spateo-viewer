try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from typing import Optional

from pyvista import BasePlotter
from trame.widgets import html, vuetify

from stviewer.assets import icon_manager, local_dataset_manager
from stviewer.interactive_viewer.pv_pipeline import Viewer

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


def toolbar_widgets(
    server,
    plotter: BasePlotter,
):
    """
    Generate standard widgets for ToolBar.

    Args:
        server: The trame server.
        plotter: The PyVista plotter to connect with the UI.
    """
    viewer = Viewer(plotter=plotter, server=server)

    # Whether to save the image
    button(
        # Must use single-quote string for JS here
        click=f"utils.download('spateo_viewer_screenshot.png', trigger('{viewer.SCREENSHOT}'), 'image/png')",
        icon="mdi-file-png-box",
        tooltip="Save screenshot",
    )
    # Whether to show the main model
    checkbox(
        model=(viewer.SHOW_MAIN_MODEL, True),
        icons=("mdi-eye-outline", "mdi-eye-off-outline"),
        tooltip=f"Toggle main model visibility ({{{{ {viewer.SHOW_MAIN_MODEL} ? 'True' : 'False' }}}})",
    )

    # Change the selection mode
    vuetify.VSpacer()
    with vuetify.VBtnToggle(v_model=(viewer.PICKING_MODE, "hover"), dense=True):
        with vuetify.VBtn(value=("item.value",), v_for="item, idx in modes"):
            vuetify.VIcon("{{item.icon}}")

    # Whether to toggle the theme between light and dark
    vuetify.VDivider(vertical=True, classes="mx-1")
    checkbox(
        model="$vuetify.theme.dark",
        icons=("mdi-lightbulb-off-outline", "mdi-lightbulb-outline"),
        tooltip=f"Toggle theme",
    )
    # Whether to toggle the background color between white and black
    checkbox(
        model=(viewer.BACKGROUND, False),
        icons=("mdi-palette-swatch-outline", "mdi-palette-swatch"),
        tooltip=f"Toggle background ({{{{ {viewer.BACKGROUND} ? 'white' : 'black' }}}})",
    )


def ui_standard_toolbar(
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
        mode: The UI view mode. Options are:

            * ``'trame'``: Uses a view that can switch between client and server rendering modes.
            * ``'server'``: Uses a view that is purely server rendering.
            * ``'client'``: Uses a view that is purely client rendering (generally safe without a virtual frame buffer)
        default_server_rendering: Whether to use server-side or client-side rendering on-start when using the ``'trame'`` mode.
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
        tb.dense = True
        tb.clipped_right = True
        toolbar_widgets(server=server, plotter=plotter,)