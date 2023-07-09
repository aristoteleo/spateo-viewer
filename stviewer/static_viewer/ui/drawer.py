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


def standard_pc_card():
    _get_default_cmap()
    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                label="Scalars",
                v_model=("pc_scalars_value", "None"),
                type="str",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
        with vuetify.VCol(cols="6"):
            vuetify.VSelect(
                v_model=("pc_picking_group", None),
                items=("Object.keys(pc_scalars_raw)",),
                show_size=True,
                dense=True,
                outlined=True,
                hide_details=True,
                classes="pt-1",
                label="Picking Group",
            )
    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="4"):
            vuetify.VCheckbox(
                v_model=("pc_add_legend", True),
                label="Add Legend",
                on_icon="mdi-view-grid-plus",
                off_icon="mdi-view-grid",
                dense=True,
                hide_details=True,
                classes="pt-1",
            )
        with vuetify.VCol(cols="4"):
            vuetify.VCheckbox(
                v_model=("pc_overwrite", False),
                label="Add Group",
                on_icon="mdi-plus-thick",
                off_icon="mdi-close-thick",
                dense=True,
                hide_details=True,
                classes="pt-1",
            )
        with vuetify.VCol(cols="4"):
            vuetify.VCheckbox(
                v_model=("pc_reload", False),
                label="Reload Model",
                on_icon="mdi-restore",
                off_icon="mdi-restore",
                dense=True,
                hide_details=True,
                classes="pt-1",
            )

    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VSelect(
                label="Coords",
                v_model=("pc_coords_value", "spatial"),
                items=(["spatial", "umap"],),
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
        with vuetify.VCol(cols="6"):
            vuetify.VSelect(
                label="Matrices",
                v_model=("pc_matrix_value", "X"),
                items=("matrices_list",),
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )

    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                label="Color",
                v_model=("pc_color_value", "None"),
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
                v_model=("pc_colormap_value", "default_cmap"),
                items=("colormaps", ["default_cmap"] + plt.colormaps()),
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
    # Opacity
    vuetify.VSlider(
        v_model=("pc_opacity_value", 1.0),
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
        v_model=("pc_ambient_value", 0.2),
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
        v_model=("pc_point_size_value", 8),
        min=0,
        max=20,
        step=1,
        label="Point Size",
        classes="mt-1",
        hide_details=True,
        dense=True,
    )


def standard_mesh_card():
    _get_default_cmap()
    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                label="Color",
                v_model=("mesh_color_value", "gainsboro"),
                type="str",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
        with vuetify.VCol(cols="6"):
            vuetify.VSelect(
                label="Style",
                v_model=("mesh_style_value", "surface"),
                items=(f"styles", ["surface", "points", "wireframe"]),
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
    # Opacity
    vuetify.VSlider(
        v_model=("mesh_opacity_value", 0.6),
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
        v_model=("mesh_ambient_value", 0.2),
        min=0,
        max=1,
        step=0.01,
        label="Ambient",
        classes="mt-1",
        hide_details=True,
        dense=True,
    )
    vuetify.VCheckbox(
        v_model=("mesh_morphology", False),
        label="Model Morphological Metrics",
        on_icon="mdi-pencil-ruler",
        off_icon="mdi-ruler",
        dense=True,
        hide_details=True,
        classes="mt-1",
    )


def standard_morphogenesis_card():
    _get_default_cmap()
    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="12"):
            vuetify.VCheckbox(
                v_model=("cal_morphogenesis", False),
                label="Calculate the Morphogenesis",
                on_icon="mdi-transit-detour",
                off_icon="mdi-transit-skip",
                dense=True,
                hide_details=True,
                classes="pt-1",
            )

    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="12"):
            vuetify.VFileInput(
                v_model=("morpho_target_anndata_path", None),
                label="Target Anndata of Morphogenesis",
                dense=True,
                outlined=True,
                hide_details=True,
                classes="pt-1",
                accept=".h5ad",
                __properties=["accept"],
            )

    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                v_model=("morpho_mapping_factor", 0.001),
                label="Mapping Factor",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                v_model=("morphofield_factor", 3000),
                label="Morphofield Factor",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                v_model=("morphopath_t_end", 10000),
                label="Morphopath Length",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                v_model=("morphopath_downsampling", 500),
                label="Morphopath Sampling",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VCheckbox(
                v_model=("morphofield_visibile", False),
                label="Morphofield Visibility",
                on_icon="mdi-pyramid",
                off_icon="mdi-pyramid-off",
                dense=True,
                hide_details=True,
                classes="pt-1",
            )
        with vuetify.VCol(cols="6"):
            vuetify.VCheckbox(
                v_model=("morphopath_visibile", False),
                label="Morphopath Visibility",
                on_icon="mdi-octahedron",
                off_icon="mdi-octahedron-off",
                dense=True,
                hide_details=True,
                classes="pt-1",
            )

    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="12"):
            vuetify.VTextField(
                v_model=("morphopath_animation_path", None),
                label="Morphogenesis Animation Output (MP4)",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )


def standard_output_card():
    with vuetify.VCardTitle(
        "Screenshot Generation",
        classes="white--text text--darken-3",
        hide_details=True,
        dense=True,
    ):
        with vuetify.VRow(classes="pt-2", dense=True):
            with vuetify.VCol(cols="12"):
                vuetify.VTextField(
                    v_model=("screenshot_path", None),
                    label="Screenshot Output (PNG or PDF)",
                    hide_details=True,
                    dense=True,
                    outlined=True,
                    classes="pt-1",
                )

    with vuetify.VCardTitle(
        "Animation Generation",
        classes="white--text text--darken-3",
        hide_details=True,
        dense=True,
    ):
        with vuetify.VRow(classes="pt-2", dense=True):
            with vuetify.VCol(cols="6"):
                vuetify.VTextField(
                    v_model=("animation_npoints", 50),
                    label="Animation N Points",
                    hide_details=True,
                    dense=True,
                    outlined=True,
                    classes="pt-1",
                )
            with vuetify.VCol(cols="6"):
                vuetify.VTextField(
                    v_model=("animation_framerate", 10),
                    label="Animation Framerate",
                    hide_details=True,
                    dense=True,
                    outlined=True,
                    classes="pt-1",
                )

        with vuetify.VRow(classes="pt-2", dense=True):
            with vuetify.VCol(cols="12"):
                vuetify.VTextField(
                    v_model=("animation_path", None),
                    label="Animation Output (MP4)",
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

        if state.active_ui.startswith("PC"):
            state.pc_scalars_value = None
            state.pc_matrix_value = "X"
            state.pc_coords_value = "spatial"
            state.pc_opacity_value = 1.0
            state.pc_ambient_value = 0.2
            state.pc_color_value = "None"
            state.pc_colormap_value = "Set3_r"
            state.pc_add_legend = False
            state.pc_overwrite = False

            state.cal_morphogenesis = False
            state.morpho_target_anndata_path = None
            state.morpho_mapping_factor = 0.001
            state.morphofield_factor = 3000
            state.morphopath_t_end = 10000
            state.morphopath_downsampling = 500
            state.morphopath_animation_path = None
            state.morphopath_predicted_models = None
            state.morphofield_visibile = False
            state.morphopath_visibile = True
        else:
            state.mesh_opacity_value = 0.6
            state.mesh_ambient_value = 0.2
            state.mesh_color_value = "gainsboro"
            state.mesh_style_value = "surface"
            state.mesh_morphology = False
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
                "Active Model",
                classes="white lighten-1 py-1 grey--text text--darken-3",
                style="user-select: none; cursor: pointer",
                hide_details=True,
                dense=True,
            )
            # PC
            with vuetify.VCardText(
                classes="py-2", v_show=f"active_model_type === 'PC'"
            ):
                with vuetify.VTabs(v_model=("pc_active_tab", 0), left=True):
                    vuetify.VTab(
                        f"Point Cloud",
                        style="width: 50%;",
                    )
                    vuetify.VTab(
                        f"Morphogenesis",
                        style="width: 50%;",
                    )
                with vuetify.VTabsItems(
                    value=("pc_active_tab",),
                    style="width: 100%; height: 100%;",
                ):
                    with vuetify.VTabItem(value=(0,)):
                        standard_pc_card()
                    with vuetify.VTabItem(value=(1,)):
                        standard_morphogenesis_card()
            # Mesh
            with vuetify.VCardText(
                classes="py-2", v_show=f"active_model_type === 'Mesh'"
            ):
                with vuetify.VTabs(v_model=("mesh_active_tab", 0), left=True):
                    vuetify.VTab(
                        f"Mesh Model",
                        style="width: 100%;",
                    )
                with vuetify.VTabsItems(
                    value=("mesh_active_tab",),
                    style="width: 100%; height: 100%;",
                ):
                    with vuetify.VTabItem(value=(0,)):
                        standard_mesh_card()

        with vuetify.VCard():
            vuetify.VCardTitle(
                "Output Widgets",
                classes="white lighten-1 py-1 grey--text text--darken-3",
                style="user-select: none; cursor: pointer",
                hide_details=True,
                dense=True,
            )
            standard_output_card()
