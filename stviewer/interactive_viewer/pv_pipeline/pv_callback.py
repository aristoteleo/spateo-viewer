import tempfile

from trame.app.file_upload import ClientFile
from vtkmodules.vtkFiltersCore import vtkThreshold
from vtkmodules.vtkFiltersGeneral import vtkExtractSelectedFrustum
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

    def __init__(self, server, plotter):
        """Initialize Viewer."""
        state, ctrl = server.state, server.controller
        self._server = server
        self._ctrl = ctrl
        self._state = state
        self._plotter = plotter

        # State variable names
        self.PICKING_MODE = f"pickingMode"
        self.SELECTION = f"selectData"
        self.UPLOAD_FILE = f"upload_file_path"

        # controller
        self._state.change(self.PICKING_MODE)(self.on_update_picking_mode)
        self._state.change(self.SELECTION)(self.on_update_selection)
        self._state.change(self.UPLOAD_FILE)(self.on_upload_file)

    @vuwrap
    def on_update_picking_mode(self, **kwargs):
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
            self._state.interactorSettings = (
                VIEW_SELECT if mode == "select" else VIEW_INTERACT
            )
            self._state.update(
                {
                    "frustrum": None,
                    "selection": None,
                    "selectData": None,
                }
            )

    @vuwrap
    def on_update_selection(self, **kwargs):
        active_model = self._plotter.actors["activeModel"].mapper.dataset
        selectData = self._state[self.SELECTION]
        if selectData is None:
            return

        extract = vtkExtractSelectedFrustum()
        extract.SetInputData(active_model)

        threshold = vtkThreshold()
        threshold.SetInputConnection(extract.GetOutputPort())
        # SetUpperThreshold(0): Remove the selected area; SetLowerThreshold(0): keep the selected area
        threshold.SetUpperThreshold(0)
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
        extract.ShowBoundsOff()
        extract.PreserveTopologyOn()
        threshold.Update()

        self._plotter.add_mesh(threshold.GetOutput(), name="activeModel")
        self._state.frustrum = vtk_mesh(extract.GetOutput())
        self._state.selection = vtk_mesh(threshold.GetOutput())
        self._state.activeModel = vtk_mesh(
            threshold.GetOutput(),
            point_arrays=[key for key in self._state.scalarParameters.keys()],
        )
        self._state.scalar = self._state.scalar
        self._state.selectData = None
        self._state.pickingMode = None

    @vuwrap
    def on_reload_main_model(self, **kwargs):
        """Reload the main model to replace the artificially adjusted active model"""
        main_model = self._plotter.actors["mainModel"].mapper.dataset.copy()
        self._plotter.add_mesh(main_model, name="activeModel")
        self._state.activeModel = vtk_mesh(
            main_model,
            point_arrays=[key for key in self._state.scalarParameters.keys()],
        )

    @vuwrap
    def on_upload_file(self, **kwargs):
        """Upload file to update the main model"""
        if self._state[self.UPLOAD_FILE] is None:
            return

        file = ClientFile(self._state[self.UPLOAD_FILE])
        if file.content:
            with tempfile.NamedTemporaryFile(suffix=file.name) as path:
                with open(path.name, "wb") as f:
                    f.write(file.content)

                from .pv_models import init_models

                main_model, active_model, init_scalar, pdd, cdd = init_models(
                    plotter=self._plotter, model_path=path.name
                )
                self._state.scalar = init_scalar
                self._state.scalarParameters = {**pdd, **cdd}
                self._state.mainModel = vtk_mesh(
                    main_model,
                    point_arrays=[key for key in pdd.keys()],
                    cell_arrays=[key for key in cdd.keys()],
                )
                self._state.activeModel = vtk_mesh(
                    main_model,
                    point_arrays=[key for key in pdd.keys()],
                    cell_arrays=[key for key in cdd.keys()],
                )
