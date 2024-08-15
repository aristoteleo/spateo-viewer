import matplotlib.colors as mcolors

# Init parameters
init_card_parameters = {
    "show_anndata_card": False,
    "show_model_card": True,
    "show_output_card": True,
}
init_pc_parameters = {
    "pc_obs_value": None,
    "pc_gene_value": None,
    "pc_scalars_raw": {"None": "None"},
    "pc_matrix_value": "X",
    "pc_coords_value": "spatial",
    "pc_opacity_value": 1.0,
    "pc_ambient_value": 0.2,
    "pc_color_value": None,
    "pc_colormap_value": "Spectral",
    "pc_point_size_value": 4,
    "pc_add_legend": False,
    "pc_picking_group": None,
    "pc_overwrite": False,
    "pc_reload": False,
    "pc_colors_list": [c for c in mcolors.CSS4_COLORS.keys()],
}
init_mesh_parameters = {
    "mesh_opacity_value": 0.3,
    "mesh_ambient_value": 0.2,
    "mesh_color_value": "gainsboro",
    "mesh_style_value": "surface",
    "mesh_morphology": False,
    "mesh_colors_list": [c for c in mcolors.CSS4_COLORS.keys()],
}
init_morphogenesis_parameters = {
    "cal_morphogenesis": False,
    "morpho_target_anndata_path": None,
    "morpho_uploaded_target_anndata_path": None,
    "morpho_mapping_method": "GP",
    "morpho_mapping_device": "cpu",
    "morpho_mapping_factor": 0.2,
    "morphofield_factor": 3000,
    "morphopath_t_end": 10000,
    "morphopath_downsampling": 500,
    "morphofield_visibile": False,
    "morphopath_visibile": False,
    "morphopath_predicted_models": None,
    "morphopath_animation_path": None,
}
init_interpolation_parameters = {
    "cal_interpolation": False,
    "interpolation_device": "cpu",
}
init_output_parameters = {
    "screenshot_path": None,
    "animation_path": None,
    "animation_npoints": 50,
    "animation_framerate": 10,
}

# costum init parameters
init_custom_parameters = {
    "custom_func": False,
    "custom_analysis": False,
    "custom_model": None,
    "custom_model_visible": False,
    "custom_parameter1": "X",
    "custom_parameter2": "recipe_monocle",
    "custom_parameter3": "pca",
    "custom_parameter4": "umap",
    "custom_parameter5": False,
    "custom_parameter6": "None",
    "custom_parameter7": 30,
    "custom_parameter8": 30,
    "custom_parameter9": "None",
    "custom_parameter10": 1,
}
