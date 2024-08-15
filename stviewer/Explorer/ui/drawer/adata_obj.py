from trame.widgets import vuetify


def anndata_object_content():
    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="12"):
            vuetify.VTextarea(
                v_model=("anndata_info.anndata_structure",),
                label="AnnData Structure",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )


def anndata_panel():
    with vuetify.VToolbar(
        dense=True, outlined=True, classes="pa-0 ma-0", style="flex: none;"
    ):
        vuetify.VIcon("mdi-apps")
        vuetify.VCardTitle(
            " AnnData Object",
            classes="pa-0 ma-0",
            style="flex: none;",
            hide_details=True,
            dense=True,
        )

        vuetify.VSpacer()
        with vuetify.VBtn(
            small=True,
            icon=True,
            click="show_anndata_card = !show_anndata_card",
        ):
            vuetify.VIcon("mdi-unfold-less-horizontal", v_if=("show_anndata_card",))
            vuetify.VIcon("mdi-unfold-more-horizontal", v_if=("!show_anndata_card",))

    # Main content
    with vuetify.VCard(style="flex: none;", classes="pa-0 ma-0"):
        with vuetify.VCardText(classes="py-2", v_if=("show_anndata_card",)):
            anndata_object_content()
