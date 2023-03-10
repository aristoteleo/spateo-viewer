from typing import Optional, Union

import matplotlib as mpl
import pyvista as pv
from pyvista import Plotter, PolyData, UnstructuredGrid

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def create_plotter(
    window_size: tuple = (1024, 1024), background: str = "black", **kwargs
) -> Plotter:
    """
    Create a plotting object to display pyvista/vtk model.

    Args:
        window_size: Window size in pixels. The default window_size is ``[1024, 768]``.
        background: The background color of the window.

    Returns:
        plotter: The plotting object to display pyvista/vtk model.
    """

    # Create an initial plotting object.
    plotter = pv.Plotter(
        window_size=window_size, off_screen=True, lighting="light_kit", **kwargs
    )

    # Set the background color of the active render window.
    plotter.background_color = background
    return plotter


def add_single_model(
    plotter: Plotter,
    model: Union[PolyData, UnstructuredGrid],
    key: Optional[str] = None,
    cmap: Optional[str] = "rainbow",
    color: Optional[str] = "gainsboro",
    ambient: float = 0.2,
    opacity: float = 1.0,
    model_style: Literal["points", "surface", "wireframe"] = "surface",
    model_size: float = 3.0,
):
    """
    Add model(s) to the plotter.
    Args:
        plotter: The plotting object to display pyvista/vtk model.
        model: A reconstructed model.
        key: The key under which are the labels.
        cmap: Name of the Matplotlib colormap to use when mapping the model.
        color: Name of the Matplotlib color to use when mapping the model.
        ambient: When lighting is enabled, this is the amount of light in the range of 0 to 1 (default 0.0) that reaches
                 the actor when not directed at the light source emitted from the viewer.
        opacity: Opacity of the model.
                 If a single float value is given, it will be the global opacity of the model and uniformly applied
                 everywhere, elif a numpy.ndarray with single float values is given, it
                 will be the opacity of each point. - should be between 0 and 1.
                 A string can also be specified to map the scalars range to a predefined opacity transfer function
                 (options include: 'linear', 'linear_r', 'geom', 'geom_r').
        model_style: Visualization style of the model. One of the following:
                     * ``model_style = 'surface'``,
                     * ``model_style = 'wireframe'``,
                     * ``model_style = 'points'``.
        model_size: If ``model_style = 'points'``, point size of any nodes in the dataset plotted.
                    If ``model_style = 'wireframe'``, thickness of lines.
    """

    if model_style == "points":
        render_spheres, render_tubes, smooth_shading = True, False, True
    elif model_style == "wireframe":
        render_spheres, render_tubes, smooth_shading = False, True, False
    else:
        render_spheres, render_tubes, smooth_shading = False, False, True
    mesh_kwargs = dict(
        scalars=key if key in model.array_names else None,
        style=model_style,
        render_points_as_spheres=render_spheres,
        render_lines_as_tubes=render_tubes,
        point_size=model_size,
        line_width=model_size,
        ambient=ambient,
        opacity=opacity,
        smooth_shading=smooth_shading,
        show_scalar_bar=False,
        cmap=cmap,
        color=color,
    )
    actor = plotter.add_mesh(model, **mesh_kwargs)
    return actor
