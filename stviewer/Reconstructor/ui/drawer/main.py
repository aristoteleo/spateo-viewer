def _get_spateo_cmap():
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap

    if "spateo_cmap" not in mpl.colormaps():
        colors = ["#4B0082", "#800080", "#F97306", "#FFA500", "#FFD700", "#FFFFCB"]
        nodes = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        mpl.colormaps.register(
            LinearSegmentedColormap.from_list("spateo_cmap", list(zip(nodes, colors)))
        )
    return "spateo_cmap"


def ui_drawer(server, layout):
    """
    Generate standard Drawer for Spateo UI.

    Args:
        server: The trame server.
        layout: The layout object.
    """

    _get_spateo_cmap()
    with layout.drawer as dr:
        # Active model
        from .model_point import pc_card_panel

        pc_card_panel()
        # Slices alignment
        from .alignment import align_card_panel

        align_card_panel()
        # Mesh reconstruction
        from .reconstruction import mesh_card_panel

        mesh_card_panel()
        # Custom
        if server.state.custom_func is True:
            from .custom_card import custom_card_panel

            custom_card_panel()
