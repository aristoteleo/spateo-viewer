from .alignment import align_card_panel
from .model_point import pc_card_panel
from .reconstruction import mesh_card_panel


def _get_default_cmap():
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap

    if "default_cmap" not in mpl.colormaps():
        colors = ["#4B0082", "#800080", "#F97306", "#FFA500", "#FFD700", "#FFFFCB"]
        nodes = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        mpl.colormaps.register(
            LinearSegmentedColormap.from_list("default_cmap", list(zip(nodes, colors)))
        )
    return "default_cmap"


def ui_drawer(layout):
    """
    Generate standard Drawer for Spateo UI.

    Args:
        layout: The layout object.
    """

    _get_default_cmap()
    with layout.drawer as dr:
        # Active model
        pc_card_panel()
        # Slices alignment
        align_card_panel()
        # Mesh reconstruction
        mesh_card_panel()
