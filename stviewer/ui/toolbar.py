try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from pyvista import BasePlotter
from trame.widgets import vuetify, html

from typing import Optional
from ..pv_pipeline import Viewer, drosophila_actors, vuwrap
from .utils import button, checkbox
from ..assets import icon_manager, local_dataset_manager
from ..dataset import sample_dataset


# -----------------------------------------------------------------------------
# GUI- UI title
# -----------------------------------------------------------------------------


def ui_title(
    layout,
    title_name="SPATEO VIEWER",
    title_icon: Optional[str] = None,
    **kwargs
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
    layout.title.style = "font-family:arial; font-size:25px; font-weight: 550; color: gray;"

    # Update the toolbar's icon
    if not (title_icon is None):
        with layout.icon as icon:
            icon.style = "margin-left: 10px;" # "width: 7vw; height: 7vh;"
            html.Img(src=title_icon, height=40, **kwargs)


# -----------------------------------------------------------------------------
# GUI- standard ToolBar
# -----------------------------------------------------------------------------


def toolbar_widgets(
    server,
    plotter: BasePlotter,
    mode: Literal["trame", "server", "client"] = "trame",
    default_server_rendering: bool = True,
):
    """
    Generate standard widgets for ToolBar.

    Args:
        server: The trame server.
        plotter: The PyVista plotter to connect with the UI.
        mode: The UI view mode. Options are:

            * ``'trame'``: Uses a view that can switch between client and server rendering modes.
            * ``'server'``: Uses a view that is purely server rendering.
            * ``'client'``: Uses a view that is purely client rendering (generally safe without a virtual frame buffer)
        default_server_rendering: Whether to use server-side or client-side rendering on-start when using the ``'trame'`` mode.
    """
    if mode != "trame":
        default_server_rendering = mode == "server"

    viewer = Viewer(plotter=plotter, server=server, suppress_rendering=mode == "client")

    # Select local directory
    vuetify.VSpacer()
    button(
        # Must use single-quote string for JS here
        click=server.controller.open_directory,
        icon="mdi-file-document-outline",
        tooltip="Select directory",
    )
    # Whether to save the image
    button(
        # Must use single-quote string for JS here
        click=f"utils.download('screenshot.png', trigger('{viewer.SCREENSHOT}'), 'image/png')",
        icon="mdi-file-png-box",
        tooltip="Save screenshot",
    )

    # Whether to toggle the theme between light and dark
    vuetify.VDivider(vertical=True, classes="mx-1")
    checkbox(
        model="$vuetify.theme.dark",
        icons=("mdi-lightbulb-off-outline", "mdi-lightbulb-outline"),
        tooltip=f"Toggle theme",
    )
    # Server rendering options
    if mode == "trame":
        checkbox(
            model=(viewer.SERVER_RENDERING, default_server_rendering),
            icons=("mdi-lan-connect", "mdi-lan-disconnect"),
            tooltip=f"Toggle rendering mode ({{{{ {viewer.SERVER_RENDERING} ? 'remote' : 'local' }}}})",
        )
    # Whether to add outline
    checkbox(
        model=(viewer.OUTLINE, False),
        icons=("mdi-cube", "mdi-cube-off"),
        tooltip=f"Toggle bounding box ({{{{ {viewer.OUTLINE} ? 'on' : 'off' }}}})",
    )
    # Whether to add grid
    checkbox(
        model=(viewer.GRID, False),
        icons=("mdi-ruler-square", "mdi-ruler-square"),
        tooltip=f"Toggle ruler ({{{{ {viewer.GRID} ? 'on' : 'off' }}}})",
    )
    # Whether to add axis legend
    checkbox(
        model=(viewer.AXIS, False),
        icons=("mdi-axis-arrow-info", "mdi-axis-arrow-info"),
        tooltip=f"Toggle axis ({{{{ {viewer.AXIS} ? 'on' : 'off' }}}})",
    )

    # Reset camera
    vuetify.VDivider(vertical=True, classes="mx-1")
    button(
        click=viewer.view_isometric,
        icon="mdi-axis-arrow",
        tooltip="Perspective view",
    )
    button(
        click=viewer.view_yz,
        icon="mdi-axis-x-arrow",
        tooltip="Reset camera X",
    )
    button(
        click=viewer.view_xz,
        icon="mdi-axis-y-arrow",
        tooltip="Reset camera Y",
    )
    button(
        click=viewer.view_xy,
        icon="mdi-axis-z-arrow",
        tooltip="Reset camera Z",
    )


class SwitchModels:
    """Callbacks for toolbar based on pyvista."""

    def __init__(self, server, plotter):
        """Initialize SwitchModels."""
        state, ctrl = server.state, server.controller
        self._server = server
        self._ctrl = ctrl
        self._state = state
        self._init_plotter = plotter

        # State variable names
        self.SELECT_SAMPLES = "select_samples"
        self.PLOTTER = "plotter"
        self.ADATA = "adata"
        self.UPLOAD_DIR = f"upload_dir"

        # Listen to state changes
        self._state.change(self.SELECT_SAMPLES)(self.on_dataset_change)

    @vuwrap
    def on_dataset_change(self, **kwargs):
        if self._state[self.SELECT_SAMPLES] in ["none", "None", None]:
            plotter = self._init_plotter
        else:
            print(self._state[self.SELECT_SAMPLES])
            print(local_dataset_manager[self._state[self.SELECT_SAMPLES]])
            (
                adata,
                pc_models,
                pc_model_ids,
                mesh_models,
                mesh_model_ids,
            ) = sample_dataset(path=local_dataset_manager[self._state[self.SELECT_SAMPLES]])
            plotter, actors, actor_names, tree = drosophila_actors(
                pc_models=pc_models,
                pc_model_ids=pc_model_ids,
                mesh_models=mesh_models,
                mesh_model_ids=mesh_model_ids,
            )

        self._state[self.PLOTTER] = plotter
        self._ctrl.view_update()

    """
    @vuwrap
    def upload_dir(self, **kwargs):

        (
            adata,
            pc_models,
            pc_model_ids,
            mesh_models,
            mesh_model_ids,
        ) = sample_dataset(path=self._state[self.UPLOAD_DIR])

        plotter, actors, actor_names, tree = drosophila_actors(
            pc_models=pc_models,
            pc_model_ids=pc_model_ids,
            mesh_models=mesh_models,
            mesh_model_ids=mesh_model_ids,
        )
        self._plotter.clear_actors()
        self._plotter = plotter
        self._ctrl.view_update()"""

    def get_plotter(self):
        if self._state[self.PLOTTER] is None:
            self._state[self.PLOTTER] = self._init_plotter
        return self._state[self.PLOTTER]

    """def get_adata(self):
        if self._state[self.ADATA] is None:
            self._state[self.ADATA] = self._init_adata
        return self._state[self.ADATA]"""


def switch_model(
    server,
    plotter: BasePlotter,
):
    """vuetify.VTextField(
        label="Upload Directory",
        v_model=(SM.UPLOAD_DIR, None),
        type="str",
        dense=True,
        outlined=True,
        hide_details=True,
        classes="ml-8",
        prepend_inner_icon="mdi-file-document-outline",
        style="max-width: 250px;",
        # filled=True,
        # rounded=True,
    )"""

    vuetify.VSpacer()
    SM = SwitchModels(server=server, plotter=plotter)
    vuetify.VSelect(
        label="Select Samples",
        v_model=(SM.SELECT_SAMPLES, "drosophila_E7_9h"),
        items=("select_samples", [key for key in local_dataset_manager.get_assets().keys()]),
        dense=True,
        outlined=True,
        hide_details=True,
        classes="ml-8",
        prepend_inner_icon="mdi-magnify",
        style="max-width: 300px;",
        # filled=True,
        rounded=True,
    )
    return SM.get_plotter()


def ui_standard_toolbar(
    server,
    layout,
    plotter: BasePlotter,
    mode: Literal["trame", "server", "client"] = "trame",
    default_server_rendering: bool = True,
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

        switch_model(server=server, plotter=plotter)
        toolbar_widgets(server=server, plotter=plotter, mode=mode, default_server_rendering=default_server_rendering)

    return plotter

























