import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from trame.app.file_upload import ClientFile
from vtkmodules.vtkFiltersCore import vtkThreshold
from vtkmodules.vtkFiltersGeneral import vtkExtractSelectedFrustum
from vtkmodules.web.utils import mesh as vtk_mesh

from .init_parameters import (
    init_active_parameters,
    init_align_parameters,
    init_mesh_parameters,
    init_picking_parameters,
    init_setting_parameters,
)

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
        self.SLICES_ALIGNMENT = "slices_alignment"
        self.PICKING_GROUP = f"picking_group"
        self.OVERWRITE = f"overwrite"
        self.OUTPUT_PATH_AM = f"activeModel_output"
        self.OUTPUT_PATH_MESH = f"mesh_output"
        self.OUTPUT_PATH_ADATA = f"anndata_output"

        # controller
        self._state.change(self.PICKING_MODE)(self.on_update_picking_mode)
        self._state.change(self.SELECTION)(self.on_update_selection)
        self._state.change(self.UPLOAD_ANNDATA)(self.on_upload_anndata)
        self._state.change(self.SLICES_ALIGNMENT)(self.on_slices_alignment)
        self._state.change("slices_key")(self.on_slices_alignment)
        self._state.change("slices_align_method")(self.on_align_method_change)
        self._state.change("slices_align_device")(self.on_slices_alignment)
        self._state.change("slices_align_factor")(self.on_slices_alignment)
        self._state.change("slices_align_max_iter")(self.on_slices_alignment)
        self._state.change("reconstruct_mesh")(self.on_reconstruct_mesh)
        self._state.change("mc_factor")(self.on_reconstruct_mesh)
        self._state.change("mesh_voronoi")(self.on_reconstruct_mesh)
        self._state.change("mesh_smooth_factor")(self.on_reconstruct_mesh)
        self._state.change("mesh_scale_factor")(self.on_reconstruct_mesh)
        self._state.change("clip_pc_with_mesh")(self.on_clip_pc_model)
        self._state.change(self.PICKING_GROUP)(self.on_picking_pc_model)
        self._state.change(self.OVERWRITE)(self.on_picking_pc_model)
        self._state.change(self.OUTPUT_PATH_AM)(self.on_download_active_model)
        self._state.change(self.OUTPUT_PATH_MESH)(self.on_download_mesh_model)
        self._state.change(self.OUTPUT_PATH_ADATA)(self.on_download_anndata)

        # Custom controller
        if self._state.custom_func is True:
            self._state.change("reconstruct_custom_model")(self.on_custom_callback)
            self._state.change("custom_parameter1")(self.on_custom_callback)
            self._state.change("custom_parameter2")(self.on_custom_callback)

    ##########################
    # Selecting active model #
    ##########################

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
    def on_picking_pc_model(self, **kwargs):
        """Picking the part of active model based on the scalar"""
        if not (self._state[self.PICKING_GROUP] in ["none", "None", None]):
            main_model = self._plotter.actors["mainModel"].mapper.dataset.copy()

            raw_labels = self._state.scalarParameters[self._state.scalar]["raw_labels"]
            if "None" in raw_labels.keys():
                if self._state[self.PICKING_GROUP] in np.unique(
                    main_model.point_data[self._state.scalar]
                ):
                    custom_picking_group = self._state[self.PICKING_GROUP]
                    added_active_model = main_model.extract_points(
                        main_model.point_data[self._state.scalar]
                        == float(custom_picking_group)
                    )
            else:
                if self._state[self.PICKING_GROUP] in raw_labels.keys():
                    custom_picking_group = raw_labels[self._state[self.PICKING_GROUP]]
                    added_active_model = main_model.extract_points(
                        main_model.point_data[self._state.scalar]
                        == float(custom_picking_group)
                    )
            if self._state[self.OVERWRITE] is True:
                active_model = self._plotter.actors["activeModel"].mapper.dataset.copy()
                active_model = active_model.merge(added_active_model)
            else:
                active_model = added_active_model
            self._plotter.add_mesh(active_model, name="activeModel")
            self._state.activeModel = vtk_mesh(
                active_model,
                point_arrays=[key for key in self._state.scalarParameters.keys()],
            )

    @vuwrap
    def on_reload_main_model(self, **kwargs):
        """Reload the main model to replace the artificially adjusted active model"""
        main_model = self._plotter.actors["mainModel"].mapper.dataset.copy()
        self._plotter.add_mesh(main_model, name="activeModel")
        self._state.activeModel = vtk_mesh(
            main_model,
            point_arrays=[key for key in self._state.scalarParameters.keys()],
        )

    #############
    # Alignment #
    #############

    @vuwrap
    def on_align_method_change(self, **kwargs):
        if str(self._state.slices_align_method) == "Paste":
            self._state.slices_align_factor = 0.1
        elif str(self._state.slices_align_method) == "Morpho":
            self._state.slices_align_factor = 20

    @vuwrap
    def on_slices_alignment(self, **kwargs):
        """Slices alignment based on the anndata of active point cloud model"""
        if self._state[self.SLICES_ALIGNMENT] is True:
            try:
                import torch
            except ImportError:
                raise ImportError(
                    "You need to install the package `torch`."
                    "\nInstall torch via `pip install torch`."
                )
            try:
                import ot
            except ImportError:
                raise ImportError(
                    "You need to install the package `POT`."
                    "\nInstall POT via `pip install POT`."
                )

            if self._state[self.UPLOAD_ANNDATA] is None:
                download_anndata_path = self._state.init_anndata
                adata_object = ad.read_h5ad(download_anndata_path)
            else:
                if type(self._state[self.UPLOAD_ANNDATA]) is dict:
                    file = ClientFile(self._state[self.UPLOAD_ANNDATA])
                    with tempfile.NamedTemporaryFile(suffix=file.name) as path:
                        with open(path.name, "wb") as f:
                            f.write(file.content)
                        adata_object = ad.read_h5ad(path.name)
                else:
                    adata_object = ad.read_h5ad(self._state[self.UPLOAD_ANNDATA])
            if str(self._state.slices_key) in adata_object.obs_keys():
                active_model = self._plotter.actors["activeModel"].mapper.dataset.copy()
                _obs_index = active_model.point_data["obs_index"]

                slices_labels = {
                    j: i
                    for i, j in self._state.scalarParameters[self._state.slices_key][
                        "raw_labels"
                    ].items()
                }
                slices_names = np.unique(
                    [
                        slices_labels[i]
                        for i in active_model.point_data[self._state.slices_key]
                    ]
                )
                slices_names.sort()

                slices_list = []
                for sn in slices_names:
                    subadata = adata_object[
                        adata_object.obs[self._state.slices_key].values == sn, :
                    ].copy()
                    subadata = subadata[subadata.obs.index.isin(_obs_index), :]
                    subadata = subadata[
                        subadata.X.sum(axis=1) != 0, subadata.X.sum(axis=0) != 0
                    ]
                    _spatial = subadata.obsm["spatial"].copy()
                    del (
                        subadata.uns,
                        subadata.var,
                        subadata.obsp,
                        subadata.varm,
                        subadata.layers,
                        subadata.obsm,
                    )
                    subadata.obs = subadata.obs[[self._state.slices_key]]
                    subadata.obsm["spatial"] = _spatial[:, :2]
                    subadata.obsm["z_spatial"] = _spatial[:, 2]
                    slices_list.append(subadata)

                _device = str(self._state.slices_align_device).lower()
                _device = (
                    "0" if _device == "gpu" and torch.cuda.is_available() else "cpu"
                )

                if str(self._state.slices_align_method) == "Paste":
                    from .pv_alignment import paste_align

                    aligned_slice = paste_align(
                        models=slices_list,
                        spatial_key="spatial",
                        key_added="align_spatial",
                        alpha=float(self._state.slices_align_factor),
                        numItermax=int(self._state.slices_align_max_iter),
                        device=_device,
                    )
                else:
                    from .pv_alignment import morpho_align

                    aligned_slice = morpho_align(
                        models=slices_list,
                        spatial_key="spatial",
                        key_added="align_spatial",
                        max_outlier_variance=int(self._state.slices_align_factor),
                        max_iter=int(self._state.slices_align_max_iter),
                        device=_device,
                    )
                aligned_spatial = [
                    pd.DataFrame(
                        np.c_[slice.obsm["align_spatial"], slice.obsm["z_spatial"]],
                        index=slice.obs.index,
                    )
                    for slice in aligned_slice
                ]
                del aligned_slice
                aligned_spatial = pd.concat(aligned_spatial, axis=0, ignore_index=False)
                aligned_spatial = aligned_spatial.loc[_obs_index, :].values
                active_model.points = aligned_spatial
                self._plotter.add_mesh(active_model, name="activeModel")
                self._state.activeModel = vtk_mesh(
                    active_model,
                    point_arrays=[key for key in self._state.scalarParameters.keys()],
                )

    ##################
    # Reconstruction #
    ##################

    @vuwrap
    def on_reconstruct_mesh(self, **kwargs):
        """Reconstruct the mesh model based on the active point cloud model"""
        if self._state.reconstruct_mesh is True:
            from .pv_tdr import construct_surface

            active_model = self._plotter.actors["activeModel"].mapper.dataset.copy()
            if active_model.n_points > 100000:
                np.random.seed(19491001)
                sampling = np.random.choice(
                    np.asarray(active_model.point_data["obs_index"]),
                    size=100000,
                    replace=False,
                )
                pc_model = active_model.extract_points(
                    np.isin(np.asarray(active_model.point_data["obs_index"]), sampling)
                )
            else:
                pc_model = active_model

            reconstructed_mesh_model = construct_surface(
                pc_model,
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

    #########
    # INPUT #
    #########

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

        self._state.update(init_active_parameters)
        self._state.update(init_picking_parameters)
        self._state.update(init_align_parameters)
        self._state.update(init_mesh_parameters)
        self._state.update(init_setting_parameters)
        self._state.update(
            {
                # main model
                "mainModel": vtk_mesh(
                    main_model,
                    point_arrays=None if len(pdd) == 0 else [key for key in pdd.keys()],
                    cell_arrays=None if len(cdd) == 0 else [key for key in cdd.keys()],
                ),
                # active model
                "activeModel": vtk_mesh(
                    main_model,
                    point_arrays=None if len(pdd) == 0 else [key for key in pdd.keys()],
                    cell_arrays=None if len(cdd) == 0 else [key for key in cdd.keys()],
                ),
                "scalar": init_scalar,
                "scalarParameters": {**pdd, **cdd},
            }
        )
        self._server.js_call(ref="render", method="resetCamera")

    ##########
    # OUTPUT #
    ##########

    @vuwrap
    def on_download_active_model(self, **kwargs):
        """Download the active model."""
        if not (self._state[self.OUTPUT_PATH_AM] in ["none", "None", None]):
            if str(self._state[self.OUTPUT_PATH_AM]).endswith(".vtk"):
                Path("stv_model").mkdir(parents=True, exist_ok=True)
                active_model = self._plotter.actors["activeModel"].mapper.dataset.copy()
                active_model.save(
                    filename=f"stv_model/{self._state[self.OUTPUT_PATH_AM]}",
                    binary=True,
                    texture=None,
                )

    @vuwrap
    def on_download_mesh_model(self, **kwargs):
        """Download the reconstructed mesh model."""
        if not (self._state[self.OUTPUT_PATH_MESH] in ["none", "None", None]):
            if str(self._state[self.OUTPUT_PATH_MESH]).endswith(".vtk"):
                if not (self._state.meshModel is None):
                    Path("stv_model").mkdir(parents=True, exist_ok=True)
                    reconstructed_mesh_model = self._plotter.actors[
                        "meshModel"
                    ].mapper.dataset.copy()
                    reconstructed_mesh_model.save(
                        filename=f"stv_model/{self._state[self.OUTPUT_PATH_MESH]}",
                        binary=True,
                        texture=None,
                    )

    @vuwrap
    def on_download_anndata(self, **kwargs):
        """Download the anndata object of active model"""
        if not (self._state[self.OUTPUT_PATH_ADATA] in ["none", "None", None]):
            if str(self._state[self.OUTPUT_PATH_ADATA]).endswith(".h5ad"):
                Path("stv_model").mkdir(parents=True, exist_ok=True)

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
                        download_adata_object = ad.read_h5ad(
                            self._state[self.UPLOAD_ANNDATA]
                        )

                active_model = self._plotter.actors["activeModel"].mapper.dataset.copy()
                _obs_index = active_model.point_data["obs_index"]
                download_adata_object = download_adata_object[_obs_index, :]
                download_adata_object.write_h5ad(
                    f"stv_model/{self._state[self.OUTPUT_PATH_ADATA]}",
                    compression="gzip",
                )

    ####################
    # Custom Callbacks #
    ####################

    @vuwrap
    def on_custom_callback(self, **kwargs):
        """Reconstruct the backbone model based on the active point cloud model"""
        if self._state.custom_func is True:
            if self._state.reconstruct_custom_model is True:
                from .pv_backbone import construct_backbone

                pc_model = self._plotter.actors["activeModel"].mapper.dataset.copy()
                custom_model = construct_backbone(
                    model=pc_model,
                    spatial_key=None,
                    nodes_key="nodes",
                    rd_method=str(self._state.custom_parameter1),
                    num_nodes=int(self._state.custom_parameter2),
                )
                custom_model.cell_data["Default"] = np.ones(
                    shape=(custom_model.n_cells, 1)
                )
                self._plotter.add_mesh(custom_model, name="customModel")
                self._plotter.actors["customModel"].prop.SetRepresentationToWireframe()
                self._state.customModel = vtk_mesh(
                    custom_model,
                    point_arrays=["nodes"],
                    cell_arrays=["Default"],
                )
