import io
import os

import matplotlib.colors as mc
import numpy as np
import pyvista as pv

from stviewer.assets import local_dataset_manager
from stviewer.assets.dataset_acquisition import abstract_anndata, sample_dataset
from vtkmodules.web.utils import mesh as vtk_mesh

VIEW_INTERACT = [
    {"button": 1, "action": "Rotate"},
    {"button": 2, "action": "Pan"},
    {"button": 3, "action": "Zoom", "scrollEnabled": True},
    {"button": 1, "action": "Pan", "alt": True},
    {"button": 1, "action": "Zoom", "control": True},
    {"button": 1, "action": "Pan", "shift": True},
    {"button": 1, "action": "Roll", "alt": True, "shift": True},
]

VIEW_SELECT = [{"button": 1, "action": "Select"}]


# -----------------------------------------------------------------------------
# Common Callback-ToolBar&Container
# -----------------------------------------------------------------------------


def vuwrap(func):
    """Call view_update in trame to synchronize changes to a view."""

    def wrapper(self, *args, **kwargs):
        ret = func(self, *args, **kwargs)
        # self._ctrl.view_update()
        return ret

    return wrapper


class Viewer:
    """Callbacks for toolbar&container based on pyvista."""

    def __init__(self, plotter, server, suppress_rendering=False):
        """Initialize Viewer."""
        state, ctrl = server.state, server.controller
        self._server = server
        self._ctrl = ctrl
        self._state = state

        self.plotter = plotter
        self.plotter.suppress_rendering = suppress_rendering
        self.active_actor = [value for value in self.plotter.actors.values()][0]
        self.main_model = self.active_actor.mapper.dataset

        # State variable names
        self.SHOW_MAIN_MODEL = f"{plotter._id_name}_show_main_model"
        self.PICKING_MODE = f"pickingMode"
        self.SELECTION = f"selectData"
        self.BACKGROUND = f"{plotter._id_name}_background"
        self.SCREENSHOT = f"{plotter._id_name}_download_screenshot"

        # controller
        ctrl.get_render_window = lambda: self.plotter.render_window

        # Listen to state changes
        self._state.change(self.SHOW_MAIN_MODEL)(self.on_show_main_model_change)
        self._state.change(self.PICKING_MODE)(self.update_picking_mode)
        self._state.change(self.SELECTION)(self.update_selection)
        self._state.change(self.BACKGROUND)(self.on_background_change)
        # Listen to events
        self._ctrl.trigger(self.SCREENSHOT)(self.screenshot)

    @vuwrap
    def on_show_main_model_change(self, **kwargs):
        """Toggle main model visibility."""
        self.active_actor.SetVisibility(self._state[self.SHOW_MAIN_MODEL])

    @vuwrap
    def update_picking_mode(self, **kwargs):
        mode = self._state[self.PICKING_MODE]
        if mode is None:
            self._state.update(
                {
                    "tooltip": "",
                    "tooltipStyle": {"display": "none"},
                    "coneVisibility": False,
                    "interactorSettings": VIEW_INTERACT,
                }
            )
        else:
            self._state.interactorSettings = VIEW_SELECT if mode == "select" else VIEW_INTERACT
            self._state.update(
                {
                    "frustrum": None,
                    "selection": None,
                    "selectData": None,
                }
            )

    @vuwrap
    def update_selection(self, **kwargs):
        selectData = self._state[self.SELECTION]
        if selectData is None:
            return

        from vtkmodules.vtkFiltersCore import vtkThreshold
        from vtkmodules.vtkFiltersGeneral import vtkExtractSelectedFrustum
        extract = vtkExtractSelectedFrustum()
        extract.SetInputData(self.main_model)

        threshold = vtkThreshold()
        threshold.SetInputConnection(extract.GetOutputPort())
        threshold.SetLowerThreshold(0)
        threshold.SetInputArrayToProcess(0, 0, 0, 1, "vtkInsidedness")

        frustrum = selectData.get("frustrum")
        vtk_frustrum = []
        for xyz in frustrum:
            vtk_frustrum += xyz
            vtk_frustrum += [1]

        extract.CreateFrustum(vtk_frustrum)
        extract.ShowBoundsOn()
        extract.PreserveTopologyOff()
        extract.Update()
        self._state.frustrum = vtk_mesh(extract.GetOutput())
        extract.ShowBoundsOff()
        extract.PreserveTopologyOn()
        threshold.Update()
        self._state.selection = vtk_mesh(threshold.GetOutput())
        self._state.selectData = None
        self._state.pickingMode = None

    @vuwrap
    def on_background_change(self, **kwargs):
        """Update background color."""
        if self._state[self.BACKGROUND]:
            self.plotter.background_color = "white"
        else:
            self.plotter.background_color = "black"

    @vuwrap
    def screenshot(self):
        """Take screenshot and add attachament."""
        self.plotter.render()
        buffer = io.BytesIO()
        self.plotter.screenshot(filename=buffer)
        buffer.seek(0)
        return self._server.protocol.addAttachment(memoryview(buffer.read()))
