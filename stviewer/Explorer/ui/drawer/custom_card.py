from trame.widgets import vuetify


def custom_card_content():
    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VCheckbox(
                v_model=("custom_analysis", False),
                label="Custom analysis calculation",
                on_icon="mdi-transit-detour",
                off_icon="mdi-transit-skip",
                dense=True,
                hide_details=True,
                classes="pt-1",
            )
        with vuetify.VCol(cols="6"):
            vuetify.VCheckbox(
                v_model=("custom_model_visible", False),
                label="Custom model visibility",
                on_icon="mdi-pyramid",
                off_icon="mdi-pyramid-off",
                dense=True,
                hide_details=True,
                classes="pt-1",
            )

    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VSelect(
                v_model=("custom_parameter1", "X"),
                items=("matrices_list",),
                label="Custom parameter 1",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
        with vuetify.VCol(cols="6"):
            vuetify.VSelect(
                v_model=("custom_parameter2", "recipe_monocle"),
                items=(["False", "recipe_monocle", "pearson_residuals"],),
                label="Custom parameter 2",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )

    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                v_model=("custom_parameter3", "pca"),
                label="Custom parameter 3",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                v_model=("custom_parameter4", "umap"),
                label="Custom parameter 4",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )

    with vuetify.VRow(classes="pt-2", dense=True):
        vuetify.VCheckbox(
            v_model=("custom_parameter5", False),
            label="Custom parameter 5",
            on_icon="mdi-plus-thick",
            off_icon="mdi-close-thick",
            dense=True,
            hide_details=True,
            classes="pt-1",
        )
        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                v_model=("custom_parameter6", "None"),
                label="Custom parameter 6",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )

    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                v_model=("custom_parameter7", 30),
                label="Custom parameter 7",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                v_model=("custom_parameter8", 30),
                label="Custom parameter 8",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )

    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                v_model=("custom_parameter9", "None"),
                label="Custom parameter 9",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                v_model=("custom_parameter10", 1),
                label="Custom parameter 10",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
