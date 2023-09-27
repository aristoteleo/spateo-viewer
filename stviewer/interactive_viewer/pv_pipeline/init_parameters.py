import pyautogui

# Init parameters
init_active_parameters = {
    "picking_group": None,
    "overwrite": False,
    "activeModelVisible": True,
    "activeModel_output": None,
    "anndata_output": None,
}
init_align_parameters = {
    "slices_alignment": False,
    "slices_key": "slices",
    "slices_align_device": "CPU",
    "slices_align_method": "Paste",
    "slices_align_factor": 0.1,
    "slices_align_max_iter": 200,
}
init_mesh_parameters = {
    "meshModel": None,
    "meshModelVisible": False,
    "reconstruct_mesh": False,
    "mc_factor": 1.0,
    "mesh_voronoi": 20000,
    "mesh_smooth_factor": 2000,
    "mesh_scale_factor": 1.0,
    "clip_pc_with_mesh": False,
    "mesh_output": None,
}
init_picking_parameters = {
    "modes": [
        {"value": "hover", "icon": "mdi-magnify"},
        {"value": "click", "icon": "mdi-cursor-default-click-outline"},
        {"value": "select", "icon": "mdi-select-drag"},
    ],
    "pickData": None,
    "selectData": None,
    "resetModel": False,
    "tooltip": "",
}
init_setting_parameters = {
    "show_active_card": True,
    "show_align_card": True,
    "show_mesh_card": True,
    "background_color": "[0, 0, 0]",
    "pixel_ratio": pyautogui.size()[0] / 500,
}
