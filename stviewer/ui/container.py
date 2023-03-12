try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from pyvista import BasePlotter
from pyvista.trame import PyVistaLocalView, PyVistaRemoteLocalView, PyVistaRemoteView
from trame.widgets import vuetify

from .toolbar import Viewer

# -----------------------------------------------------------------------------
# GUI- standard Container
# -----------------------------------------------------------------------------


def ui_standard_container(
    server,
    layout,
    plotter: BasePlotter,
    mode: Literal["trame", "server", "client"] = "trame",
    default_server_rendering: bool = True,
    **kwargs,
):
    """
    Generate standard VContainer for Spateo UI.

    Args:
        server: The trame server.
        layout: The layout object.
        plotter: The PyVista plotter to connect with the UI.
        mode: The UI view mode. Options are:

            * ``'trame'``: Uses a view that can switch between client and server rendering modes.
            * ``'server'``: Uses a view that is purely server rendering.
            * ``'client'``: Uses a view that is purely client rendering (generally safe without a virtual frame buffer)
        default_server_rendering: Whether to use server-side or client-side rendering on-start when using the ``'trame'`` mode.
        kwargs: Additional parameters that will be passed to ``pyvista.trame.app.PyVistaXXXXView`` function.
    """
    if mode != "trame":
        default_server_rendering = mode == "server"

    viewer = Viewer(plotter, server, suppress_rendering=mode == "client")
    ctrl = server.controller

    with layout.content as con:
        with vuetify.VContainer(
            fluid=True,
            classes="pa-0 fill-height",
        ):
            if mode == "trame":
                view = PyVistaRemoteLocalView(
                    plotter,
                    mode=(
                        # Must use single-quote string for JS here
                        f"{viewer.SERVER_RENDERING} ? 'remote' : 'local'",
                        "remote" if default_server_rendering else "local",
                    ),
                    **kwargs,
                )
                ctrl.view_update_image = view.update_image
            elif mode == "server":
                view = PyVistaRemoteView(plotter, **kwargs)
            elif mode == "client":
                view = PyVistaLocalView(plotter, **kwargs)

            ctrl.view_update = view.update
            ctrl.view_reset_camera = view.reset_camera
            ctrl.view_push_camera = view.push_camera
            ctrl.on_server_ready.add(view.update)
    return view
