from trame.widgets import vuetify


def custom_card_content():
    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VCheckbox(
                v_model=("reconstruct_custom_model", False),
                label="Reconstruct Custom Model",
                on_icon="mdi-billiards-rack",
                off_icon="mdi-dots-triangle",
                dense=True,
                hide_details=True,
                classes="pt-1",
            )
        with vuetify.VCol(cols="6"):
            vuetify.VCheckbox(
                v_model=("customModelVisible", False),
                label="Visibility of Custom Model",
                on_icon="mdi-eye-outline",
                off_icon="mdi-eye-off-outline",
                dense=True,
                hide_details=True,
                classes="pt-1",
            )

    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VSelect(
                label="Custom Parameter 1",
                v_model=("custom_parameter1", "ElPiGraph"),
                items=(["ElPiGraph", "SimplePPT", "PrinCurve"],),
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )

        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                label="Custom Parameter 2",
                v_model=("custom_parameter2", 50),
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )

    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="12"):
            vuetify.VTextField(
                v_model=("custom_model_output", None),
                label="Custom model Output",
                show_size=True,
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )


def custom_card_panel():
    with vuetify.VToolbar(
        dense=True, outlined=True, classes="pa-0 ma-0", style="flex: none;"
    ):
        vuetify.VIcon("mdi-billiards-rack")
        vuetify.VCardTitle(
            " Custom Card",
            classes="pa-0 ma-0",
            style="flex: none;",
            hide_details=True,
            dense=True,
        )

        vuetify.VSpacer()
        with vuetify.VBtn(
            small=True,
            icon=True,
            click="show_custom_card = !show_custom_card",
        ):
            vuetify.VIcon("mdi-unfold-less-horizontal", v_if=("show_custom_card",))
            vuetify.VIcon("mdi-unfold-more-horizontal", v_if=("!show_custom_card",))

    # Main content
    with vuetify.VCard(style="flex: none;", classes="pa-0 ma-0"):
        with vuetify.VCardText(classes="py-2", v_if=("show_custom_card",)):
            custom_card_content()
