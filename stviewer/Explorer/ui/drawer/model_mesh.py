from trame.widgets import vuetify


def mesh_card_content():
    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VCombobox(
                label="Color",
                v_model=("mesh_color_value", "None"),
                items=("mesh_colors_list",),
                type="str",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
        with vuetify.VCol(cols="6"):
            vuetify.VCombobox(
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
        v_model=("mesh_opacity_value", 0.3),
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
