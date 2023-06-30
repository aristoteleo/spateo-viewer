try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from typing import Optional

import matplotlib.pyplot as plt
from pyvista import BasePlotter
from trame.widgets import trame, vuetify

from stviewer.static_viewer.pv_pipeline import PVCB

# -----------------------------------------------------------------------------
# Card
# -----------------------------------------------------------------------------


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


def standard_pc_card(default_values: Optional[dict] = None):
    _get_default_cmap()
    _default_values = {
        "layer": "X_counts",
        "scalars": "None",
        "point_size": 8,
        "color": "gainsboro",
        "cmap": _get_default_cmap(),
        "opacity": 1,
        "ambient": 0.2,
        "mm_color": "gainsboro",
        "mm_cmap": _get_default_cmap(),
    }
    if not (default_values is None):
        _default_values.update(default_values)

    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                label="Scalars",
                v_model=("actor_scalars_value", _default_values["scalars"]),
                type="str",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
        with vuetify.VCol(cols="6"):
            vuetify.VSelect(
                label="Matrices",
                v_model=("actor_matrix_value", _default_values["layer"]),
                items=("matrices", ["X_counts", "X_log1p"]),
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )

    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                label="Color",
                v_model=("actor_color_value", _default_values["color"]),
                type="str",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
        # Colormap
        with vuetify.VCol(cols="6"):
            vuetify.VSelect(
                label="Colormap",
                v_model=("actor_colormap_value", _default_values["cmap"]),
                items=("colormaps", ["default_cmap"] + plt.colormaps()),
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
    # Opacity
    vuetify.VSlider(
        v_model=("actor_opacity_value", _default_values["opacity"]),
        min=0,
        max=1,
        step=0.01,
        label="Opacity",
        classes="mt-1",
        hide_details=True,
        dense=True,
    )
    # Ambient
    vuetify.VSlider(
        v_model=("actor_ambient_value", _default_values["ambient"]),
        min=0,
        max=1,
        step=0.01,
        label="Ambient",
        classes="mt-1",
        hide_details=True,
        dense=True,
    )
    # Point size
    vuetify.VSlider(
        v_model=("actor_point_size_value", _default_values["point_size"]),
        min=0,
        max=20,
        step=1,
        label="Point Size",
        classes="mt-1",
        hide_details=True,
        dense=True,
    )
    with vuetify.VRow(classes="pt-2", dense=True):
        # Whether to show vectorfield model
        with vuetify.VCol(cols="6"):
            vuetify.VCheckbox(
                v_model=("show_vectorpc", False),
                on_icon="mdi-pyramid",
                off_icon="mdi-pyramid-off",
                dense=True,
                hide_details=True,
                label="VectorField",
                classes="pt-1",
            )
        # Whether to show trajectory model
        with vuetify.VCol(cols="6"):
            vuetify.VCheckbox(
                v_model=("show_trajectory", False),
                on_icon="mdi-octahedron",
                off_icon="mdi-octahedron-off",
                dense=True,
                hide_details=True,
                label="Trajectory",
                classes="pt-1",
            )
    with vuetify.VRow(classes="pt-2", dense=True):
        # MM Color
        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                label="MM Color",
                v_model=("mm_actor_color_value", _default_values["mm_color"]),
                type="str",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
        # MM Colormap
        with vuetify.VCol(cols="6"):
            vuetify.VSelect(
                label="MM Colormap",
                v_model=("mm_actor_colormap_value", _default_values["mm_cmap"]),
                items=("mm_colormaps", ["default_cmap"] + plt.colormaps()),
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )


def standard_mesh_card(default_values: Optional[dict] = None):
    _get_default_cmap()
    _default_values = {
        "style": "surface",
        "color": "gainsboro",
        "opacity": 0.5,
        "ambient": 0.2,
        "mm_color": "gainsboro",
        "mm_cmap": None,
    }
    if not (default_values is None):
        _default_values.update(default_values)

    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="12"):
            vuetify.VTextField(
                label="Color",
                v_model=("actor_color_value", _default_values["color"]),
                type="str",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
    with vuetify.VRow(classes="pt-2", dense=True):
        # Style
        with vuetify.VCol(cols="12"):
            vuetify.VSelect(
                label="Style",
                v_model=("actor_style_value", _default_values["style"]),
                items=(f"styles", ["surface", "points", "wireframe"]),
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
    # Opacity
    vuetify.VSlider(
        v_model=("actor_opacity_value", _default_values["opacity"]),
        min=0,
        max=1,
        step=0.01,
        label="Opacity",
        classes="mt-1",
        hide_details=True,
        dense=True,
    )
    # Ambient
    vuetify.VSlider(
        v_model=("actor_ambient_value", _default_values["ambient"]),
        min=0,
        max=1,
        step=0.01,
        label="Ambient",
        classes="mt-1",
        hide_details=True,
        dense=True,
    )
    with vuetify.VRow(classes="pt-2", dense=True):
        # Whether to show vectorfield model
        with vuetify.VCol(cols="6"):
            vuetify.VCheckbox(
                v_model=("show_vectormesh", False),
                on_icon="mdi-pyramid",
                off_icon="mdi-pyramid-off",
                dense=True,
                hide_details=True,
                label="Vetorfield",
                classes="pt-1",
            )
        # Whether to show trajectory model
        with vuetify.VCol(cols="6"):
            vuetify.VCheckbox(
                v_model=("show_trajectory", False),
                on_icon="mdi-octahedron",
                off_icon="mdi-octahedron-off",
                dense=True,
                hide_details=True,
                label="Trajectory",
                classes="pt-1",
            )
    with vuetify.VRow(classes="pt-2", dense=True):
        # MM Color
        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                label="MM Color",
                v_model=("mm_actor_color_value", _default_values["mm_color"]),
                type="str",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
        # MM Colormap
        with vuetify.VCol(cols="6"):
            vuetify.VSelect(
                label="MM Colormap",
                v_model=("mm_actor_colormap_value", _default_values["mm_cmap"]),
                items=("mm_colormaps", ["default_cmap"] + plt.colormaps()),
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )


# -----------------------------------------------------------------------------
# GitTree
# -----------------------------------------------------------------------------


def pipeline(server, plotter):
    """Create a vuetify GitTree."""
    state, ctrl = server.state, server.controller

    @ctrl.set("actives_change")
    def actives_change(ids):
        _id = ids[0]
        active_actor_id = state.actor_ids[int(_id) - 1]
        state.active_ui = active_actor_id
        state.active_model_type = str(state.active_ui).split("_")[0]
        state.active_id = int(_id)
        ctrl.view_update()

    # Visibility Change
    @ctrl.set("visibility_change")
    def visibility_change(event):
        _id = event["id"]
        _visibility = event["visible"]
        active_actor = [value for value in plotter.actors.values()][int(_id) - 1]
        active_actor.SetVisibility(_visibility)
        if _visibility is True:
            state.vis_ids.append(int(_id) - 1)
        else:
            state.vis_ids.remove(int(_id) - 1)
        state.vis_ids = list(set(state.vis_ids))
        ctrl.view_update()

    trame.GitTree(
        sources=("pipeline",),
        actives_change=(ctrl.actives_change, "[$event]"),
        visibility_change=(ctrl.visibility_change, "[$event]"),
    )


# -----------------------------------------------------------------------------
# GUI-standard Drawer
# -----------------------------------------------------------------------------


def ui_standard_drawer(
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

    PVCB(server=server, plotter=plotter, suppress_rendering=mode == "client")
    with layout.drawer as dr:
        pipeline(server=server, plotter=plotter)
        vuetify.VDivider(classes="mb-2")
        with vuetify.VCard():
            vuetify.VCardTitle(
                "{{ active_ui }}",
                classes="white lighten-1 py-1 grey--text text--darken-3",
                style="user-select: none; cursor: pointer",
                hide_details=True,
                dense=True,
            )
            with vuetify.VCardText(
                classes="py-2", v_show=f"active_model_type === 'PC'"
            ):
                standard_pc_card()
            with vuetify.VCardText(
                classes="py-2", v_show=f"active_model_type === 'Mesh'"
            ):
                standard_mesh_card()
