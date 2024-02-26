import matplotlib.pyplot as plt
from trame.widgets import vuetify


def pc_card_content():
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
                label="Add picked Group",
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
                items=("pc_colormaps",),
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
