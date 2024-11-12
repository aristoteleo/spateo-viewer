import matplotlib.pyplot as plt
from trame.widgets import vuetify


def pc_card_content():
    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VCombobox(
                label="Observation Annotation",
                v_model=("pc_obs_value", None),
                items=("available_obs",),
                type="str",
                show_size=True,
                dense=True,
                outlined=True,
                hide_details=True,
                classes="pt-1",
            )
        with vuetify.VCol(cols="6"):
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
            vuetify.VCombobox(
                v_model=("pc_picking_group", None),
                items=("Object.keys(pc_scalars_raw)",),
                label="Picking Group",
                show_size=True,
                dense=True,
                outlined=True,
                hide_details=True,
                classes="pt-1",
            )
        with vuetify.VCol(cols="6"):
            vuetify.VCheckbox(
                v_model=("pc_overwrite", False),
                label="Add Picked Group",
                on_icon="mdi-plus-thick",
                off_icon="mdi-close-thick",
                dense=True,
                hide_details=True,
                classes="pt-1",
            )

    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VCombobox(
                label="Available Genes",
                v_model=("pc_gene_value", None),
                items=("available_genes",),
                type="str",
                show_size=True,
                dense=True,
                outlined=True,
                hide_details=True,
                classes="pt-1",
            )
        with vuetify.VCol(cols="6"):
            vuetify.VCheckbox(
                v_model=("pc_add_legend", True),
                label="Add Legend",
                on_icon="mdi-view-grid-plus",
                off_icon="mdi-view-grid",
                dense=True,
                hide_details=True,
                classes="pt-1",
            )
    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                v_model=("interpolation_device", "cpu"),
                label="Interpolation Device",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
        with vuetify.VCol(cols="6"):
            vuetify.VCheckbox(
                v_model=("cal_interpolation", True),
                label="GP Interpolation",
                on_icon="mdi-smoke-detector",
                off_icon="mdi-smoke-detector-off",
                dense=True,
                hide_details=True,
                classes="pt-1",
            )

    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VCombobox(
                v_model=("pc_coords_value", "spatial"),
                items=("anndata_info.anndata_obsm_keys",),
                label="Coords",
                type="str",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
        with vuetify.VCol(cols="6"):
            vuetify.VCombobox(
                v_model=("pc_matrix_value", "X"),
                items=("anndata_info.anndata_metrices",),
                label="Matrices",
                type="str",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )

    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VCombobox(
                v_model=("pc_color_value", "None"),
                items=("pc_colors_list",),
                label="Color",
                type="str",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
        # Colormap
        with vuetify.VCol(cols="6"):
            vuetify.VCombobox(
                v_model=("pc_colormap_value", "Spectral"),
                items=("pc_colormaps_list",),
                label="Colormap",
                type="str",
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
        step=0.05,
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
        step=0.05,
        label="Ambient",
        classes="mt-1",
        hide_details=True,
        dense=True,
    )
    # Point size
    vuetify.VSlider(
        v_model=("pc_point_size_value", 2),
        min=0,
        max=20,
        step=1,
        label="Point Size",
        classes="mt-1",
        hide_details=True,
        dense=True,
    )
