from trame.widgets import vuetify


def pc_card_content():
    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VSelect(
                v_model=("scalar", "Default"),
                items=("Object.values(scalarParameters)",),
                show_size=True,
                dense=True,
                outlined=True,
                hide_details=True,
                classes="pt-1",
                label="Scalars",
            )
        with vuetify.VCol(cols="6"):
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
                label="Colormap",
            )
    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VSelect(
                v_model=("picking_group", None),
                items=("Object.keys(scalarParameters[scalar].raw_labels)",),
                show_size=True,
                dense=True,
                outlined=True,
                hide_details=True,
                classes="pt-2",
                label="Picking Group",
            )
        with vuetify.VCol(cols="6"):
            vuetify.VCheckbox(
                v_model=("overwrite", False),
                label="Overwrite the Active Model",
                on_icon="mdi-plus-thick",
                off_icon="mdi-close-thick",
                dense=True,
                hide_details=True,
                classes="pt-1",
            )

    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="12"):
            vuetify.VTextField(
                v_model=("activeModel_output", None),
                label="Active Model Output",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="12"):
            vuetify.VTextField(
                v_model=("anndata_output", None),
                label="Anndata Output",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="12"):
            vuetify.VCheckbox(
                v_model=("activeModelVisible", True),
                label="Visibility of Active Model",
                on_icon="mdi-eye-outline",
                off_icon="mdi-eye-off-outline",
                dense=True,
                hide_details=True,
                classes="pt-1",
            )


def pc_card_panel():
    with vuetify.VToolbar(
        dense=True, outlined=True, classes="pa-0 ma-0", style="flex: none;"
    ):
        vuetify.VIcon("mdi-format-paint")
        vuetify.VCardTitle(
            " Active Model",
            classes="pa-0 ma-0",
            style="flex: none;",
            hide_details=True,
            dense=True,
        )

        vuetify.VSpacer()
        with vuetify.VBtn(
            small=True,
            icon=True,
            click="show_active_card = !show_active_card",
        ):
            vuetify.VIcon("mdi-unfold-less-horizontal", v_if=("show_active_card",))
            vuetify.VIcon("mdi-unfold-more-horizontal", v_if=("!show_active_card",))

    # Main content
    with vuetify.VCard(style="flex: none;", classes="pa-0 ma-0"):
        with vuetify.VCardText(classes="py-2", v_if=("show_active_card",)):
            pc_card_content()
