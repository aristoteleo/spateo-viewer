from trame.widgets import vuetify


def morphogenesis_card_content():
    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="12"):
            vuetify.VCheckbox(
                v_model=("cal_morphogenesis", False),
                label="Calculate the Morphogenesis",
                on_icon="mdi-transit-detour",
                off_icon="mdi-transit-skip",
                dense=True,
                hide_details=True,
                classes="pt-1",
            )

    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="12"):
            vuetify.VFileInput(
                v_model=("morpho_uploaded_target_anndata_path", None),
                label="Upload Target AnnData (.h5ad)",
                dense=True,
                classes="pt-1",
                accept=".h5ad",
                hide_details=True,
                show_size=True,
                small_chips=True,
                truncate_length=0,
                outlined=True,
                __properties=["accept"],
            )
    with vuetify.VRow(classes="pt-2", dense=True):
        avaliable_samples = [
            "uploaded_target_anndata",
        ]
        with vuetify.VCol(cols="12"):
            vuetify.VSelect(
                label="Target AnnData",
                v_model=("morpho_target_anndata_path", None),
                items=(avaliable_samples,),
                dense=True,
                outlined=True,
                hide_details=True,
                classes="pt-1",
            )

    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VSelect(
                v_model=("morpho_mapping_method", "GP"),
                items=(["OT", "GP"],),
                label="Mapping Method",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                v_model=("morpho_mapping_device", "cpu"),
                label="Mapping Device",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )

    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                v_model=("morpho_mapping_factor", 0.2),
                label="Mapping Factor",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                v_model=("morphofield_factor", 3000),
                label="Morphofield Factor",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                v_model=("morphopath_t_end", 10000),
                label="Morphopath Length",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
        with vuetify.VCol(cols="6"):
            vuetify.VTextField(
                v_model=("morphopath_downsampling", 500),
                label="Morphopath Sampling",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="6"):
            vuetify.VCheckbox(
                v_model=("morphofield_visibile", False),
                label="Morphofield Visibility",
                on_icon="mdi-pyramid",
                off_icon="mdi-pyramid-off",
                dense=True,
                hide_details=True,
                classes="pt-1",
            )
        with vuetify.VCol(cols="6"):
            vuetify.VCheckbox(
                v_model=("morphopath_visibile", False),
                label="Morphopath Visibility",
                on_icon="mdi-octahedron",
                off_icon="mdi-octahedron-off",
                dense=True,
                hide_details=True,
                classes="pt-1",
            )

    with vuetify.VRow(classes="pt-2", dense=True):
        with vuetify.VCol(cols="12"):
            vuetify.VTextField(
                v_model=("morphopath_animation_path", None),
                label="Morphogenesis Animation Output (MP4)",
                hide_details=True,
                dense=True,
                outlined=True,
                classes="pt-1",
            )
