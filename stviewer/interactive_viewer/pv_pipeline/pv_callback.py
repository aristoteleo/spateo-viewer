import tempfile
from pathlib import Path

import numpy as np
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
        self.UPLOAD_ANNDATA = f"upload_anndata"
        self.RECONSTRUCT_MESH = f"reconstruct_mesh"

        # controller
        self._state.change(self.PICKING_MODE)(self.on_update_picking_mode)
        self._state.change(self.SELECTION)(self.on_update_selection)
        self._state.change(self.UPLOAD_ANNDATA)(self.on_upload_anndata)
        self._state.change(self.RECONSTRUCT_MESH)(self.on_reconstruct_mesh)
        self._state.change("mc_factor")(self.on_reconstruct_mesh)
        self._state.change("mesh_voronoi")(self.on_reconstruct_mesh)
        self._state.change("mesh_smooth_factor")(self.on_reconstruct_mesh)
        self._state.change("mesh_scale_factor")(self.on_reconstruct_mesh)
        self._state.change("clip_pc_with_mesh")(self.on_clip_pc_model)

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
    def on_reconstruct_mesh(self, **kwargs):
        """Reconstruct the mesh model based on the active point cloud model"""
        if self._state.reconstruct_mesh is True:
            from .pv_tdr import construct_surface

            active_model = self._plotter.actors["activeModel"].mapper.dataset.copy()
            reconstructed_mesh_model = construct_surface(
                active_model,
                mc_scale_factor=float(self._state.mc_factor),
                nclus=int(self._state.mesh_voronoi),
                smooth=int(self._state.mesh_smooth_factor),
                scale_factor=float(self._state.mesh_scale_factor),
            )
            reconstructed_mesh_model.cell_data["Default"] = np.ones(
                shape=(reconstructed_mesh_model.n_cells, 1)
            )

            self._plotter.add_mesh(reconstructed_mesh_model, name="meshModel")
            self._state.meshModel = vtk_mesh(
                reconstructed_mesh_model,
                point_arrays=["Default"],
            )

    @vuwrap
    def on_clip_pc_model(self, **kwargs):
        """Clip the original pc using the reconstructed surface and reconstruct new point cloud"""
        if not (self._state.meshModel is None):
            if self._state.clip_pc_with_mesh is True:
                active_model = self._plotter.actors["activeModel"].mapper.dataset.copy()
                reconstructed_mesh_model = self._plotter.actors[
                    "meshModel"
                ].mapper.dataset.copy()

                select_pc = active_model.select_enclosed_points(
                    surface=reconstructed_mesh_model, check_surface=False
                )
                select_pc1 = select_pc.threshold(
                    0.5, scalars="SelectedPoints"
                ).extract_surface()
                select_pc2 = select_pc.threshold(
                    0.5, scalars="SelectedPoints", invert=True
                ).extract_surface()
                inside_pc = (
                    select_pc1
                    if select_pc1.n_points > select_pc2.n_points
                    else select_pc2
                )

                self._plotter.add_mesh(inside_pc, name="activeModel")
                self._state.activeModel = vtk_mesh(
                    inside_pc,
                    point_arrays=[key for key in self._state.scalarParameters.keys()],
                )

    @vuwrap
    def on_upload_anndata(self, **kwargs):
        """Upload file to update the main model"""
        if self._state[self.UPLOAD_ANNDATA] is None:
            return

        from .pv_models import init_models

        if type(self._state[self.UPLOAD_ANNDATA]) is dict:
            file = ClientFile(self._state[self.UPLOAD_ANNDATA])
            if file.content:
                with tempfile.NamedTemporaryFile(suffix=file.name) as path:
                    with open(path.name, "wb") as f:
                        f.write(file.content)
                    main_model, active_model, init_scalar, pdd, cdd = init_models(
                        plotter=self._plotter, anndata_path=path.name
                    )
        else:
            anndata_path = self._state[self.UPLOAD_ANNDATA]
            main_model, active_model, init_scalar, pdd, cdd = init_models(
                plotter=self._plotter, anndata_path=anndata_path
            )

        self._state.scalar = init_scalar
        self._state.scalarParameters = {**pdd, **cdd}
        self._state.mainModel = vtk_mesh(
            main_model,
            point_arrays=None if len(pdd) == 0 else [key for key in pdd.keys()],
            cell_arrays=None if len(cdd) == 0 else [key for key in cdd.keys()],
        )
        self._state.activeModel = vtk_mesh(
            main_model,
            point_arrays=None if len(pdd) == 0 else [key for key in pdd.keys()],
            cell_arrays=None if len(cdd) == 0 else [key for key in cdd.keys()],
        )

    @vuwrap
    def on_download_active_model(self, **kwargs):
        """Download the active model."""
        Path("stv_model").mkdir(parents=True, exist_ok=True)
        active_model = self._plotter.actors["activeModel"].mapper.dataset.copy()
        active_model.save(
            filename="stv_model/active_point_cloud_model.vtk", binary=True, texture=None
        )

    @vuwrap
    def on_download_mesh_model(self, **kwargs):
        """Download the reconstructed mesh model."""
        if not (self._state.meshModel is None):
            Path("stv_model").mkdir(parents=True, exist_ok=True)
            reconstructed_mesh_model = self._plotter.actors[
                "meshModel"
            ].mapper.dataset.copy()
            reconstructed_mesh_model.save(
                filename="stv_model/reconstructed_mesh_model.vtk",
                binary=True,
                texture=None,
            )

    @vuwrap
    def on_download_anndata(self, **kwargs):
        """Download the anndata object of active model"""
        Path("stv_model").mkdir(parents=True, exist_ok=True)

        import anndata as ad

        if self._state[self.UPLOAD_ANNDATA] is None:
            download_anndata_path = self._state.init_anndata
            download_adata_object = ad.read_h5ad(download_anndata_path)
        else:
            if type(self._state[self.UPLOAD_ANNDATA]) is dict:
                file = ClientFile(self._state[self.UPLOAD_ANNDATA])
                with tempfile.NamedTemporaryFile(suffix=file.name) as path:
                    with open(path.name, "wb") as f:
                        f.write(file.content)
                    download_adata_object = ad.read_h5ad(path.name)
            else:
                download_adata_object = ad.read_h5ad(self._state[self.UPLOAD_ANNDATA])

        active_model = self._plotter.actors["activeModel"].mapper.dataset.copy()
        _obs_index = active_model.point_data["obs_index"]
        download_adata_object = download_adata_object[_obs_index, :]
        download_adata_object.write_h5ad(
            "stv_model/active_model_anndata.h5ad", compression="gzip"
        )
