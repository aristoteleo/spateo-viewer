try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

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


def ui_container(
    server,
    layout,
):
    """
    Generate standard VContainer for Spateo UI.

    Args:
        server: The trame server.
        layout: The layout object.
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
                ref="render",
                background=(state.background_color,),
                picking_modes=("[pickingMode]",),
                interactor_settings=("interactorSettings", VIEW_INTERACT),
                click="pickData = $event",
                hover="pickData = $event",
                select="selectData = $event",
            ) as view:
                ctrl.view_reset_camera = view.reset_camera
                with vtk_widgets.VtkGeometryRepresentation(
                    id="activeModel",
                    v_if="activeModel",
                    actor=("{ visibility: activeModelVisible }",),
                    color_map_preset=("colorMap",),
                    color_data_range=("scalarParameters[scalar].range",),
                    mapper=(
                        "{ colorByArrayName: scalar, scalarMode: scalarParameters[scalar].scalarMode,"
                        " interpolateScalarsBeforeMapping: true, scalarVisibility: scalar !== 'Default' }",
                    ),
                    property=(
                        {
                            "pointSize": state.pixel_ratio,
                            "representation": 1,
                            "opacity": 1,
                            "ambient": 0.3,
                        },
                    ),
                ):
                    vtk_widgets.VtkMesh("activeModel", state=("activeModel",))
                with vtk_widgets.VtkGeometryRepresentation(
                    id="meshModel",
                    v_if="meshModel",
                    actor=("{ visibility: meshModelVisible }",),
                    property=(
                        {
                            "representation": 1,
                            "opacity": 0.6,
                            "ambient": 0.1,
                        },
                    ),
                ):
                    vtk_widgets.VtkMesh("meshModel", state=("meshModel",))

                # Custom model visibility
                if state.custom_func is True:
                    with vtk_widgets.VtkGeometryRepresentation(
                        id="customModel",
                        v_if="customModel",
                        actor=("{ visibility: customModelVisible }",),
                        property=(
                            {
                                "lineWidth": state.custom_model_size,
                                "pointSize": state.custom_model_size,
                                "representation": 1,
                                "opacity": 1,
                                "ambient": 0.1,
                            },
                        ),
                    ):
                        vtk_widgets.VtkMesh("customModel", state=("customModel",))
