try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from typing import Optional

import matplotlib.pyplot as plt
from pyvista import BasePlotter
from trame.widgets import html, trame, vuetify

from stviewer.interactive_viewer.pv_pipeline import Viewer

from .utils import button

# -----------------------------------------------------------------------------
# Card
# -----------------------------------------------------------------------------


def widgets_card():
    # Active model
    vuetify.VCardTitle(
        "Active Model",
        classes="white--text text--darken-3",
        hide_details=True,
        dense=True,
    )
    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VSelect(
                v_model=("scalar", "Default"),
                items=("Object.values(scalarParameters)",),
                show_size=True,
                # truncate_length=25,
                dense=True,
                outlined=True,
                hide_details=True,
                classes="pt-1",
                # style="max-width: 150px",
                label="Scalar of active model",
            )
        with vuetify.VCol(cols="6"):
            vuetify.VSelect(
                v_model=("picking_group", None),
                items=("Object.keys(scalarParameters[scalar].raw_labels)",),
                show_size=True,
                # truncate_length=25,
                dense=True,
                outlined=True,
                hide_details=True,
                classes="pt-1",
                # style="max-width: 150px",
                label="Picking group",
            )

    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="12"):
            vuetify.VSelect(
                v_model=("colorMap", "erdc_rainbow_bright"),
                items=("trame.utils.vtk.vtkColorPresetItems('')",),
                show_size=True,
                # truncate_length=25,
                dense=True,
                outlined=True,
                hide_details=True,
                classes="pt-1",
                # style="max-width: 150px",
                label="Colormap of active model",
            )
    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="12"):
            vuetify.VTextField(
                v_model=("activeModel_output", None),
                label="Active model output",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="12"):
            vuetify.VTextField(
                v_model=("anndata_output", None),
                label="Anndata output",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="12"):
            vuetify.VCheckbox(
                v_model=("activeModelVisible", False),
                label="Visibility of active model",
                on_icon="mdi-eye-outline",
                off_icon="mdi-eye-off-outline",
                dense=True,
                hide_details=True,
                classes="pt-1",
            )

    # Mesh model reconstruction
    vuetify.VCardTitle(
        "Mesh Reconstruction",
        classes="white--text text--darken-3",
        hide_details=True,
        dense=True,
    )
    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VCheckbox(
                v_model=("reconstruct_mesh", False),
                label="Reconstruct mesh model",
                on_icon="mdi-billiards-rack",
                off_icon="mdi-dots-triangle",
                dense=True,
                hide_details=True,
                classes="pt-1",
            )
        with vuetify.VCol(cols="6"):
            vuetify.VCheckbox(
                v_model=("clip_pc_with_mesh", False),
                label="Clip with mesh model",
                on_icon="mdi-box-cutter",
                off_icon="mdi-box-cutter-off",
                dense=True,
                hide_details=True,
                classes="pt-1",
            )

    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                v_model=("mc_factor", 1.0),
                label="MC factor",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                v_model=("mesh_voronoi", 20000),
                label="Voronoi clustering",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )

    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                v_model=("mesh_smooth_factor", 1000),
                label="Smooth factor",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                v_model=("mesh_scale_factor", 1.0),
                label="Scale factor",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )

    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="12"):
            vuetify.VTextField(
                v_model=("mesh_output", None),
                label="Reconstructed mesh output",
                show_size=True,
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="12"):
            vuetify.VCheckbox(
                v_model=("meshModelVisible", False),
                label="Visibility of mesh model",
                on_icon="mdi-eye-outline",
                off_icon="mdi-eye-off-outline",
                dense=True,
                hide_details=True,
                classes="pt-1",
            )


# -----------------------------------------------------------------------------
# GUI-standard Drawer
# -----------------------------------------------------------------------------


def ui_standard_drawer(layout):
    """
    Generate standard Drawer for Spateo UI.

    Args:
        layout: The layout object.
    """

    with layout.drawer as dr:
        with vuetify.VCard():
            vuetify.VCardTitle(
                "Model Widgets",
                classes="white lighten-1 py-1 grey--text text--darken-3",
                style="user-select: none; cursor: pointer",
                hide_details=True,
                dense=True,
            )

            with vuetify.VCardText(classes="py-2"):
                widgets_card()
