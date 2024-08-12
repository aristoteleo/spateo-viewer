from trame.widgets import vuetify


def output_screenshot_content():
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


def output_animation_content():
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


def output_panel():
    with vuetify.VToolbar(
        dense=True, outlined=True, classes="pa-0 ma-0", style="flex: none;"
    ):
        vuetify.VIcon("mdi-download")
        vuetify.VCardTitle(
            " Output Widgets",
            classes="pa-0 ma-0",
            style="flex: none;",
            hide_details=True,
            dense=True,
        )

        vuetify.VSpacer()
        with vuetify.VBtn(
            small=True,
            icon=True,
            click="show_output_card = !show_output_card",
        ):
            vuetify.VIcon("mdi-unfold-less-horizontal", v_if=("show_output_card",))
            vuetify.VIcon("mdi-unfold-more-horizontal", v_if=("!show_output_card",))

    # Main content
    with vuetify.VCard(style="flex: none;", classes="pa-0 ma-0"):
        with vuetify.VCardText(classes="py-2", v_if=("show_output_card",)):
            items = ["Screenshot", "Animation"]
            with vuetify.VTabs(v_model=("output_active_tab", 0), left=True):
                for item in items:
                    vuetify.VTab(
                        item,
                        style="width: 50%;",
                    )
            with vuetify.VTabsItems(
                value=("output_active_tab",),
                style="width: 100%; height: 100%;",
            ):
                with vuetify.VTabItem(value=(0,)):
                    output_screenshot_content()
                with vuetify.VTabItem(value=(1,)):
                    output_animation_content()
