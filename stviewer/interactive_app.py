try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from tkinter import Tk, filedialog

from .assets import icon_manager, local_dataset_manager
from .server import get_trame_server
from .static_viewer import (
    create_plotter,
    init_actors,
    ui_layout,
    ui_standard_container,
    ui_standard_drawer,
    ui_standard_toolbar,
)

# export WSLINK_MAX_MSG_SIZE=1000000000    # 1GB

# Get a Server to work with
interactive_server = get_trame_server()
state, ctrl = interactive_server.state, interactive_server.controller
state.trame__title = "SPATEO VIEWER"
state.trame__favicon = icon_manager.spateo_logo
state.setdefault("active_ui", None)

# Generate a new plotter
plotter = create_plotter()
