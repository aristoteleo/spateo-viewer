from trame.widgets import vuetify


def mesh_card_content():
    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VCheckbox(
                v_model=("reconstruct_mesh", False),
                label="Reconstruct Mesh Model",
                on_icon="mdi-billiards-rack",
                off_icon="mdi-dots-triangle",
                dense=True,
                hide_details=True,
                classes="pt-1",
            )
        with vuetify.VCol(cols="6"):
            vuetify.VCheckbox(
                v_model=("clip_pc_with_mesh", False),
                label="Clip with Mesh Model",
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
                label="MC Factor",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                v_model=("mesh_voronoi", 20000),
                label="Voronoi Clustering",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )

    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                v_model=("mesh_smooth_factor", 1000),
                label="Smooth Factor",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                v_model=("mesh_scale_factor", 1.0),
                label="Scale Factor",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )

    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="12"):
            vuetify.VTextField(
                v_model=("mesh_output", None),
                label="Reconstructed Mesh Output",
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
                label="Visibility of Mesh Model",
                on_icon="mdi-eye-outline",
                off_icon="mdi-eye-off-outline",
                dense=True,
                hide_details=True,
                classes="pt-1",
            )


def mesh_card_panel():
    with vuetify.VToolbar(
        dense=True, outlined=True, classes="pa-0 ma-0", style="flex: none;"
    ):
        vuetify.VIcon("mdi-billiards-rack")
        vuetify.VCardTitle(
            " Mesh Reconstruction",
            classes="pa-0 ma-0",
            style="flex: none;",
            hide_details=True,
            dense=True,
        )

        vuetify.VSpacer()
        with vuetify.VBtn(
            small=True,
            icon=True,
            click="show_mesh_card = !show_mesh_card",
        ):
            vuetify.VIcon("mdi-unfold-less-horizontal", v_if=("show_mesh_card",))
            vuetify.VIcon("mdi-unfold-more-horizontal", v_if=("!show_mesh_card",))

    # Main content
    with vuetify.VCard(style="flex: none;", classes="pa-0 ma-0"):
        with vuetify.VCardText(classes="py-2", v_if=("show_mesh_card",)):
            mesh_card_content()
