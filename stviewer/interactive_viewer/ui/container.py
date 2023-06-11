try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from pyvista import BasePlotter
from trame.widgets import html
from trame.widgets import vtk as vtk_widgets
from trame.widgets import vuetify

VIEW_INTERACT = [
    {"button": 1, "action": "Rotate"},
    {"button": 2, "action": "Pan"},
    {"button": 3, "action": "Zoom", "scrollEnabled": True},
    {"button": 1, "action": "Pan", "alt": True},
    {"button": 1, "action": "Zoom", "control": True},
    {"button": 1, "action": "Pan", "shift": True},
    {"button": 1, "action": "Roll", "alt": True, "shift": True},
]

VIEW_SELECT = [{"button": 1, "action": "Select"}]


# -----------------------------------------------------------------------------
# GUI- standard Container
# -----------------------------------------------------------------------------


def ui_standard_container(
    server,
    layout,
    plotter: BasePlotter,
):
    """
    Generate standard VContainer for Spateo UI.

    Args:
        server: The trame server.
        layout: The layout object.
        plotter: The PyVista plotter to connect with the UI.
        kwargs: Additional parameters that will be passed to ``pyvista.trame.app.PyVistaXXXXView`` function.
    """

    state, ctrl = server.state, server.controller
    with layout.content:
        with vuetify.VContainer(
            fluid=True, classes="pa-0 fill-height", style="position: relative;"
        ):
            with vuetify.VCard(
                style=("tooltipStyle", {"display": "none"}), elevation=2, outlined=True
            ):
                with vuetify.VCardText():
                    html.Pre("{{ tooltip }}")
            with vtk_widgets.VtkView(
                ref="view",
                background=("[0, 0, 0]",),
                picking_modes=("[pickingMode]",),
                interactor_settings=("interactorSettings", VIEW_INTERACT),
                click="pickData = $event",
                hover="pickData = $event",
                select="selectData = $event",
            ) as view:
                view.reset_camera()
                with vtk_widgets.VtkGeometryRepresentation(
                    id="activeModel",
                    actor=("{ visibility: activeModelVisible }",),
                    property=(
                        {
                            "color": [0.99, 0.13, 0.37],
                            "pointSize": state.pixel_ratio,
                            "Opacity": 1.0,
                            "Ambient": 0.2,
                            "representation": 0,
                            "RepresentationToPoints": "TRUE",
                            "RenderPointsAsSpheres": "TRUE",
                        },
                    ),
                ):
                    vtk_widgets.VtkMesh(
                        "activeModel",
                        dataset=plotter.actors["activeModel"].mapper.dataset,
                    )
