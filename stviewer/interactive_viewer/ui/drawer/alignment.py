from trame.widgets import vuetify


def align_card_content():
    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VCheckbox(
                v_model=("slices_alignment", False),
                label="Align slices",
                on_icon="mdi-layers-outline",
                off_icon="mdi-layers-off-outline",
                dense=True,
                hide_details=True,
                classes="pt-1",
            )
        with vuetify.VCol(cols="6"):
            vuetify.VSelect(
                v_model=("slices_align_method", "Paste"),
                items=(["Paste", "Morpho"],),
                show_size=True,
                dense=True,
                outlined=True,
                hide_details=True,
                classes="pt-1",
                label="Method of Alignment",
            )
    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                v_model=("slices_key", "slices"),
                label="Slices Key",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                v_model=("slices_align_factor", 0.1),
                label="Align Factor",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                v_model=("slices_align_max_iter", 200),
                label="Max Iterations",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
        with vuetify.VCol(cols="6"):
            vuetify.VSelect(
                v_model=("slices_align_device", "CPU"),
                items=(["CPU", "GPU"],),
                show_size=True,
                dense=True,
                outlined=True,
                hide_details=True,
                classes="pt-1",
                label="Device",
            )


def align_card_panel():
    with vuetify.VToolbar(
        dense=True, outlined=True, classes="pa-0 ma-0", style="flex: none;"
    ):
        vuetify.VIcon("mdi-target")
        vuetify.VCardTitle(
            " Slices Alignment",
            classes="pa-0 ma-0",
            style="flex: none;",
            hide_details=True,
            dense=True,
        )

        vuetify.VSpacer()
        with vuetify.VBtn(
            small=True,
            icon=True,
            click="show_align_card = !show_align_card",
        ):
            vuetify.VIcon("mdi-unfold-less-horizontal", v_if=("show_align_card",))
            vuetify.VIcon("mdi-unfold-more-horizontal", v_if=("!show_align_card",))

    # Main content
    with vuetify.VCard(style="flex: none;", classes="pa-0 ma-0"):
        with vuetify.VCardText(classes="py-2", v_if=("show_align_card",)):
            align_card_content()
