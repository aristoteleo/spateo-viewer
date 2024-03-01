import os
import tempfile
from pathlib import Path

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from trame.app.file_upload import ClientFile

from stviewer.assets import local_dataset_manager
from stviewer.assets.dataset_acquisition import abstract_anndata, sample_dataset

from .init_parameters import (
    init_mesh_parameters,
    init_morphogenesis_parameters,
    init_output_parameters,
    init_pc_parameters,
)
from .pv_actors import generate_actors, generate_actors_tree

# -----------------------------------------------------------------------------
# Common Callback-ToolBar&Container
# -----------------------------------------------------------------------------


def vuwrap(func):
    """Call view_update in trame to synchronize changes to a view."""

    def wrapper(self, *args, **kwargs):
        ret = func(self, *args, **kwargs)
        self._ctrl.view_update()
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

        # State variable names
        self.SHOW_MAIN_MODEL = f"{plotter._id_name}_show_main_model"
        self.BACKGROUND = f"{plotter._id_name}_background"
        self.GRID = f"{plotter._id_name}_grid_visibility"
        self.OUTLINE = f"{plotter._id_name}_outline_visibility"
        self.EDGES = f"{plotter._id_name}_edge_visibility"
        self.AXIS = f"{plotter._id_name}_axis_visiblity"
        self.SCREENSHOT = f"{plotter._id_name}_download_screenshot"
        self.SERVER_RENDERING = f"{plotter._id_name}_use_server_rendering"

        # controller
        ctrl.get_render_window = lambda: self.plotter.render_window

        # Listen to state changes
        self._state.change(self.SHOW_MAIN_MODEL)(self.on_show_main_model_change)
        self._state.change(self.BACKGROUND)(self.on_background_change)
        self._state.change(self.GRID)(self.on_grid_visiblity_change)
        self._state.change(self.OUTLINE)(self.on_outline_visiblity_change)
        self._state.change(self.EDGES)(self.on_edge_visiblity_change)
        self._state.change(self.AXIS)(self.on_axis_visiblity_change)
        self._state.change(self.SERVER_RENDERING)(self.on_rendering_mode_change)

    @vuwrap
    def on_show_main_model_change(self, **kwargs):
        """Toggle main model visibility."""
        # _id = int(self._state.active_id) - 1 if self._state.active_id != 0 else int(self._state.active_id)
        # active_actor = [value for value in self.plotter.actors.values()][_id]
        # active_actor.SetVisibility(self._state[self.SHOW_MAIN_MODEL])

        for i in self._state.vis_ids:
            actor = [value for value in self.plotter.actors.values()][i]
            actor.SetVisibility(self._state[self.SHOW_MAIN_MODEL])
        self._ctrl.view_update()

    @vuwrap
    def on_background_change(self, **kwargs):
        """Update background color."""
        if self._state[self.BACKGROUND]:
            self.plotter.background_color = "white"
        else:
            self.plotter.background_color = "black"

    @vuwrap
    def on_edge_visiblity_change(self, **kwargs):
        """Toggle edge visibility for all actors."""
        value = self._state[self.GRID]
        for _, actor in self.plotter.actors.items():
            if isinstance(actor, pv.Actor):
                actor.prop.show_edges = value

    @vuwrap
    def view_isometric(self):
        """View isometric."""
        self.plotter.view_isometric()
        self._ctrl.view_push_camera(force=True)

    @vuwrap
    def view_yz(self):
        """View YZ plane."""
        self.plotter.view_yz()
        self._ctrl.view_push_camera(force=True)

    @vuwrap
    def view_xz(self):
        """View XZ plane."""
        self.plotter.view_xz()
        self._ctrl.view_push_camera(force=True)

    @vuwrap
    def view_xy(self):
        """View XY plane."""
        self.plotter.view_xy()
        self._ctrl.view_push_camera(force=True)

    @vuwrap
    def reset_camera(self):
        """Reset the camera."""
        # self.plotter.reset_camera()
        self._ctrl.view_reset_camera(force=True)

    @vuwrap
    def on_grid_visiblity_change(self, **kwargs):
        """Handle axes grid visibility."""
        if self._state[self.GRID]:
            bg_rgb = mc.to_rgb(self.plotter.background_color.name)
            cbg_rgb = (1 - bg_rgb[0], 1 - bg_rgb[1], 1 - bg_rgb[2])
            self.plotter.show_grid(color=cbg_rgb)
        else:
            self.plotter.remove_bounds_axes()

    @vuwrap
    def on_outline_visiblity_change(self, **kwargs):
        """Handle outline visibility."""
        if self._state[self.OUTLINE]:
            bg_rgb = mc.to_rgb(self.plotter.background_color.name)
            cbg_rgb = (1 - bg_rgb[0], 1 - bg_rgb[1], 1 - bg_rgb[2])
            self.plotter.add_bounding_box(color=cbg_rgb, reset_camera=False)
        else:
            self.plotter.remove_bounding_box()

    @vuwrap
    def on_axis_visiblity_change(self, **kwargs):
        """Handle outline visibility."""
        if self._state[self.AXIS]:
            self.plotter.show_axes()
        else:
            self.plotter.hide_axes()

    @vuwrap
    def on_rendering_mode_change(self, **kwargs):
        """Handle any configurations when the render mode changes between client and server."""
        if not self._state[self.SERVER_RENDERING]:
            self._ctrl.view_push_camera(force=True)

    @property
    def actors(self):
        """Get dataset actors."""
        return {k: v for k, v in self.plotter.actors.items() if isinstance(v, pv.Actor)}


class SwitchModels:
    """Callbacks for toolbar based on pyvista."""

    def __init__(self, server, plotter):
        """Initialize SwitchModels."""
        state, ctrl = server.state, server.controller
        self._server = server
        self._ctrl = ctrl
        self._state = state
        self.plotter = plotter

        # State variable names
        self.SELECT_SAMPLES = "select_samples"
        self.MATRICES_LIST = f"matrices_list"

        # Listen to state changes
        self._state.change(self.SELECT_SAMPLES)(self.on_dataset_change)
        self._state.change(self.MATRICES_LIST)(self.on_dataset_change)

    @vuwrap
    def on_dataset_change(self, **kwargs):
        if self._state[self.SELECT_SAMPLES] is None:
            pass
        else:
            if self._state[self.SELECT_SAMPLES] == "uploaded_sample":
                path = self._state.selected_dir
            else:
                path = local_dataset_manager[self._state[self.SELECT_SAMPLES]]
            (
                adata,
                pc_models,
                pc_model_ids,
                mesh_models,
                mesh_model_ids,
                custom_colors,
            ) = sample_dataset(path=path)

            # Generate actors
            self.plotter.clear_actors()
            pc_actors, mesh_actors = generate_actors(
                plotter=self.plotter,
                pc_models=pc_models,
                pc_model_names=pc_model_ids,
                mesh_models=mesh_models,
                mesh_model_names=mesh_model_ids,
            )

            # Generate the relationship tree of actors
            actors, actor_names, actor_tree = generate_actors_tree(
                pc_actors=pc_actors,
                mesh_actors=mesh_actors,
            )

            self._state.update(
                {
                    "init_dataset": False,
                    "anndata_path": os.path.join(
                        os.path.join(path, "h5ad"),
                        os.listdir(path=os.path.join(path, "h5ad"))[0],
                    ),
                    "matrices_list": ["X"] + [i for i in adata.layers.keys()],
                    # setting
                    "actor_ids": actor_names,
                    "pipeline": actor_tree,
                    "active_id": 1,
                    "active_ui": actor_names[0],
                    "active_model_type": str(actor_names[0]).split("_")[0],
                    "vis_ids": [
                        i
                        for i, actor in enumerate(self.plotter.actors.values())
                        if actor.visibility
                    ],
                    "show_model_card": True,
                    "show_output_card": True,
                    "pc_colormaps": ["default_cmap"] + custom_colors + plt.colormaps(),
                }
            )
            self._state.update(init_pc_parameters)
            self._state.update(init_mesh_parameters)
            self._state.update(init_morphogenesis_parameters)
            self._state.update(init_output_parameters)
            self._ctrl.view_reset_camera(force=True)
            self._ctrl.view_update()


# -----------------------------------------------------------------------------
# Common Callbacks-Drawer
# -----------------------------------------------------------------------------


class PVCB:
    """Callbacks for drawer based on pyvista."""

    def __init__(self, server, plotter, suppress_rendering=False):
        """Initialize PVCB."""
        state, ctrl = server.state, server.controller

        self._server = server
        self._ctrl = ctrl
        self._state = state
        self._plotter = plotter
        self._plotter.suppress_rendering = suppress_rendering

        # State variable names
        # pc model
        self.pcSCALARS = f"pc_scalars_value"
        self.pcMATRIX = f"pc_matrix_value"
        self.pcCOORDS = f"pc_coords_value"
        self.pcOPACITY = f"pc_opacity_value"
        self.pcAMBIENT = f"pc_ambient_value"
        self.pcCOLOR = f"pc_color_value"
        self.pcCOLORMAP = f"pc_colormap_value"
        self.pcPOINTSIZE = f"pc_point_size_value"
        self.pcLEGEND = "pc_add_legend"
        self.pcPICKINGGROUP = f"pc_picking_group"
        self.pcOVERWRITE = f"pc_overwrite"
        self.pcRELOAD = f"pc_reload"
        self.adINFO = f"anndata_info"
        # mesh model
        self.meshOPACITY = f"mesh_opacity_value"
        self.meshAMBIENT = f"mesh_ambient_value"
        self.meshCOLOR = f"mesh_color_value"
        self.meshSTYLE = f"mesh_style_value"
        self.meshMORPHOLOGY = f"mesh_morphology"
        # morphogenesis
        self.morphoCALCULATION = f"cal_morphogenesis"
        self.morphoANNDATA = f"morpho_target_anndata_path"
        self.morphoUPLOADEDANNDATA = f"morpho_uploaded_target_anndata_path"
        self.morphoMAPPINGmethod = f"morpho_mapping_method"
        self.morphoMAPPINGdevice = f"morpho_mapping_device"
        self.morphoMAPPING = f"morpho_mapping_factor"
        self.morphoFIELD = f"morphofield_factor"
        self.morphoTEND = f"morphopath_t_end"
        self.morphoSAMPLING = f"morphopath_downsampling"
        self.morphoANIMATION = f"morphopath_animation_path"
        self.morphoPREDICTEDMODELS = f"morphopath_predicted_models"
        self.morphoSHOWFIELD = f"morphofield_visibile"
        self.morphoSHOWTRAJECTORY = f"morphopath_visibile"

        # output
        self.PLOTTER_SCREENSHOT = "screenshot_path"
        self.PLOTTER_ANIMATION = "animation_path"

        # Listen to state changes
        self._state.change(self.pcSCALARS)(self.on_scalars_change)
        self._state.change(self.pcSCALARS)(self.on_legend_change)
        self._state.change(self.pcMATRIX)(self.on_scalars_change)
        self._state.change(self.pcCOORDS)(self.on_coords_change)
        self._state.change(self.pcOPACITY)(self.on_opacity_change)
        self._state.change(self.pcAMBIENT)(self.on_ambient_change)
        self._state.change(self.pcCOLOR)(self.on_color_change)
        self._state.change(self.pcCOLORMAP)(self.on_colormap_change)
        self._state.change(self.pcPOINTSIZE)(self.on_point_size_change)
        self._state.change(self.pcLEGEND)(self.on_legend_change)
        self._state.change(self.pcPICKINGGROUP)(self.on_picking_pc_model)
        self._state.change(self.pcOVERWRITE)(self.on_picking_pc_model)
        self._state.change(self.pcRELOAD)(self.on_reload_main_model)
        self._state.change(self.adINFO)(self.on_show_anndata_info)

        self._state.change(self.meshOPACITY)(self.on_opacity_change)
        self._state.change(self.meshAMBIENT)(self.on_ambient_change)
        self._state.change(self.meshCOLOR)(self.on_color_change)
        self._state.change(self.meshSTYLE)(self.on_style_change)
        self._state.change(self.meshMORPHOLOGY)(self.on_morphology_change)

        self._state.change(self.morphoCALCULATION)(self.on_cal_morphogenesis)
        self._state.change(self.morphoSHOWFIELD)(self.on_show_morpho_model_change)
        self._state.change(self.morphoSHOWTRAJECTORY)(self.on_show_morpho_model_change)
        self._state.change(self.morphoANIMATION)(self.on_morphogenesis_animation)

        self._state.change(self.PLOTTER_SCREENSHOT)(self.on_plotter_screenshot)
        self._state.change(self.PLOTTER_ANIMATION)(self.on_plotter_animation)

        # Custom controller
        if self._state.custom_func is True:
            self._state.change("custom_analysis")(self.on_custom_callback)
            self._state.change("custom_model_visible")(self.on_show_custom_model)

    @vuwrap
    def on_show_anndata_info(self, **kwargs):
        if self._state[self.adINFO]:
            _active_id = (
                1 if int(self._state.active_id) == 0 else int(self._state.active_id) - 1
            )
            active_name = self._state.actor_ids[_active_id]
            active_actor = self._plotter.actors[active_name]
            _obs_index = active_actor.mapper.dataset.point_data["obs_index"]
            _adata = abstract_anndata(path=self._state.anndata_path)[_obs_index, :]

            # Anndata basic info
            obs_str, var_str, uns_str, obsm_str, layers_str = (
                f"    obs:",
                f"    var:",
                f"    uns:",
                f"    obsm:",
                f"    layers:",
            )

            if len(list(_adata.obs.keys())) != 0:
                for key in list(_adata.obs.keys()):
                    obs_str = obs_str + f" '{key}',"
            if len(list(_adata.var.keys())) != 0:
                for key in list(_adata.var.keys()):
                    var_str = var_str + f" '{key}',"
            if len(list(_adata.uns.keys())) != 0:
                for key in list(_adata.uns.keys()):
                    uns_str = uns_str + f" '{key}',"
            if len(list(_adata.obsm.keys())) != 0:
                for key in list(_adata.obsm.keys()):
                    obsm_str = obsm_str + f" '{key}',"
            if len(list(_adata.layers.keys())) != 0:
                for key in list(_adata.layers.keys()):
                    layers_str = layers_str + f" '{key}',"

            ad_info = f"AnnData object with n_obs × n_vars = {_adata.shape[0]} × {_adata.shape[1]}\n"
            for ad_str in [obs_str, var_str, uns_str, obsm_str, layers_str]:
                if ad_str.endswith(","):
                    ad_info = ad_info + f"{ad_str[:-1]}\n"

            if "anndata_info_actor" in self._plotter.actors.keys():
                self._plotter.remove_actor(self._plotter.actors["anndata_info_actor"])
            self._plotter.add_text(
                text=ad_info,
                font="arial",
                color="white",
                font_size=15,
                position="upper_left",
                name="anndata_info_actor",
            )
        else:
            if "anndata_info_actor" in self._plotter.actors.keys():
                self._plotter.remove_actor(self._plotter.actors["anndata_info_actor"])
        self._ctrl.view_update()

    @vuwrap
    def on_scalars_change(self, **kwargs):
        _active_id = (
            1 if int(self._state.active_id) == 0 else int(self._state.active_id) - 1
        )
        active_name = self._state.actor_ids[_active_id]
        active_actor = self._plotter.actors[active_name]

        if str(active_name).startswith("PC"):
            if self._state[self.pcSCALARS] in ["none", "None", None]:
                active_actor.mapper.scalar_visibility = False
                for morpho_key in ["MorphoField", "MorphoPath"]:
                    if morpho_key in self._plotter.actors.keys():
                        self._plotter.actors[
                            morpho_key
                        ].mapper.scalar_visibility = False
                self._ctrl.view_update()
            else:
                _obs_index = active_actor.mapper.dataset.point_data["obs_index"]
                _adata = abstract_anndata(path=self._state.anndata_path)[_obs_index, :]
                if self._state[self.pcSCALARS] in set(_adata.obs_keys()):
                    change_array = True
                    array = _adata.obs[self._state[self.pcSCALARS]].values
                    if array.dtype == "category":
                        array = np.asarray(array, dtype=str)
                    if np.issubdtype(array.dtype, np.number):
                        array = np.asarray(array, dtype=float)
                        self._state.pc_scalars_raw = {"None": "None"}
                    else:
                        od = {o: i for i, o in enumerate(np.unique(array))}
                        array = np.asarray(
                            list(map(lambda x: od[x], array)), dtype=float
                        )
                        self._state.pc_scalars_raw = od
                    array = array.reshape(-1, 1)
                elif self._state[self.pcSCALARS] in set(_adata.var_names.tolist()):
                    change_array = True
                    matrix_id = self._state[self.pcMATRIX]
                    self._state.pc_scalars_raw = {"None": "None"}
                    if matrix_id == "X":
                        array = np.asarray(
                            _adata[:, self._state[self.pcSCALARS]].X.sum(axis=1),
                            dtype=float,
                        )
                    else:
                        array = np.asarray(
                            _adata[:, self._state[self.pcSCALARS]]
                            .layers[matrix_id]
                            .sum(axis=1),
                            dtype=float,
                        )
                elif (
                    self._state[self.pcSCALARS]
                    in active_actor.mapper.dataset.point_data.keys()
                ):
                    array = active_actor.mapper.dataset[
                        self._state[self.pcSCALARS]
                    ].copy()
                    change_array = True
                else:
                    change_array = False

                if change_array is True:
                    active_actor.mapper.dataset.point_data[
                        self._state[self.pcSCALARS]
                    ] = array
                    active_actor.mapper.scalar_range = (
                        active_actor.mapper.dataset.get_data_range(
                            self._state[self.pcSCALARS]
                        )
                    )

                    active_actor.mapper.SelectColorArray(self._state[self.pcSCALARS])
                    active_actor.mapper.lookup_table.cmap = self._state[self.pcCOLORMAP]
                    active_actor.mapper.SetScalarModeToUsePointFieldData()
                    active_actor.mapper.scalar_visibility = True
                    active_actor.mapper.Update()
                    self._plotter.actors[active_name] = active_actor
                    self.on_legend_change()

                    for morpho_key in ["MorphoField", "MorphoPath"]:
                        if morpho_key in self._plotter.actors.keys():
                            morpho_actor = self._plotter.actors[morpho_key]
                            morpho_index = morpho_actor.mapper.dataset.point_data[
                                "obs_index"
                            ]

                            morpho_array = np.asarray(
                                pd.DataFrame(array, index=_obs_index).loc[
                                    morpho_index, 0
                                ]
                            )
                            morpho_actor.mapper.dataset.point_data[
                                self._state[self.pcSCALARS]
                            ] = morpho_array
                            morpho_actor.mapper.scalar_range = (
                                active_actor.mapper.scalar_range
                            )
                            morpho_actor.mapper.SelectColorArray(
                                self._state[self.pcSCALARS]
                            )
                            morpho_actor.mapper.lookup_table.cmap = self._state[
                                self.pcCOLORMAP
                            ]
                            morpho_actor.mapper.SetScalarModeToUsePointFieldData()
                            morpho_actor.mapper.scalar_visibility = True
                            morpho_actor.mapper.Update()
                            self._plotter.actors[morpho_key] = morpho_actor
                else:
                    active_actor.mapper.scalar_visibility = False
                    for morpho_key in ["MorphoField", "MorphoPath"]:
                        if morpho_key in self._plotter.actors.keys():
                            self._plotter.actors[
                                morpho_key
                            ].mapper.scalar_visibility = False
                self._ctrl.view_update()

    @vuwrap
    def on_legend_change(self, **kwargs):
        _active_id = (
            1 if int(self._state.active_id) == 0 else int(self._state.active_id) - 1
        )
        active_name = self._state.actor_ids[_active_id]
        active_actor = self._plotter.actors[active_name]

        if self._state[self.pcLEGEND] and active_actor.mapper.scalar_visibility:
            if len(self._plotter.scalar_bars.keys()) != 0:
                self._plotter.remove_scalar_bar()
            if self._plotter.legend:
                self._plotter.remove_legend()
            if "None" in self._state.pc_scalars_raw.keys():
                self._plotter.add_scalar_bar(
                    self._state[self.pcSCALARS],
                    mapper=active_actor.mapper,
                    bold=True,
                    interactive=False,
                    vertical=True,
                    title_font_size=30,
                    label_font_size=25,
                    outline=False,
                    fmt="%10.2f",
                )
            else:
                import matplotlib as mpl

                legend_labels = [i for i in self._state.pc_scalars_raw.keys()]

                lscmap = mpl.cm.get_cmap(self._state[self.pcCOLORMAP])
                legend_hex = [
                    mpl.colors.to_hex(lscmap(i))
                    for i in np.linspace(0, 1, len(legend_labels))
                ]

                legend_entries = [
                    [label, hex] for label, hex in zip(legend_labels, legend_hex)
                ]
                self._plotter.add_legend(
                    legend_entries,
                    face="circle",
                    bcolor=None,
                    loc="lower right",
                )
        else:
            if len(self._plotter.scalar_bars.keys()) != 0:
                self._plotter.remove_scalar_bar()
            if self._plotter.legend:
                self._plotter.remove_legend()
        self._ctrl.view_update()

    @vuwrap
    def on_picking_pc_model(self, **kwargs):
        """Picking the part of active model based on the scalar"""
        if not (self._state[self.pcPICKINGGROUP] in ["none", "None", None]):
            _active_id = (
                1 if int(self._state.active_id) == 0 else int(self._state.active_id) - 1
            )
            active_name = self._state.actor_ids[_active_id]
            if "pc_raw_model" in self._plotter.actors.keys():
                active_actor = self._plotter.actors["pc_raw_model"].copy()
                basis_model = self._plotter.actors[active_name].mapper.dataset.copy()
            else:
                active_actor = self._plotter.actors[active_name]
                self._plotter.actors["pc_raw_model"] = active_actor.copy()
                basis_model = None

            active_model = active_actor.mapper.dataset.copy()
            if "None" in self._state.pc_scalars_raw.keys():
                custom_picking_group = self._state[self.pcPICKINGGROUP]
                added_active_model = active_model.extract_points(
                    active_model.point_data[self._state[self.pcSCALARS]]
                    == float(custom_picking_group)
                )
            else:
                custom_picking_group = self._state.pc_scalars_raw[
                    self._state[self.pcPICKINGGROUP]
                ]
                added_active_model = active_model.extract_points(
                    active_model.point_data[self._state[self.pcSCALARS]]
                    == float(custom_picking_group)
                )
            if self._state[self.pcOVERWRITE] is True:
                active_model = (
                    added_active_model
                    if basis_model is None
                    else basis_model.merge(added_active_model)
                )
            else:
                active_model = added_active_model
            self._plotter.actors[active_name].mapper.dataset = active_model
            self._ctrl.view_update()

    @vuwrap
    def on_reload_main_model(self, **kwargs):
        """Reload the main model to replace the artificially adjusted active model"""
        # if self._state[self.pcRELOAD]:
        _active_id = (
            1 if int(self._state.active_id) == 0 else int(self._state.active_id) - 1
        )
        active_name = self._state.actor_ids[_active_id]
        if "pc_raw_model" in self._plotter.actors.keys():
            self._plotter.actors[active_name].mapper.dataset = self._plotter.actors[
                "pc_raw_model"
            ].mapper.dataset.copy()
            self._plotter.remove_actor(self._plotter.actors["pc_raw_model"])
            self._state[self.pcPICKINGGROUP] = "None"
            self._state[self.pcOVERWRITE] = False
        self._ctrl.view_update()

    @vuwrap
    def on_coords_change(self, **kwargs):
        _active_id = (
            1 if int(self._state.active_id) == 0 else int(self._state.active_id) - 1
        )
        active_name = self._state.actor_ids[_active_id]
        active_actor = self._plotter.actors[active_name]

        _obs_index = active_actor.mapper.dataset.point_data["obs_index"]
        _adata = abstract_anndata(path=self._state.anndata_path)[_obs_index, :]
        if str(self._state[self.pcCOORDS]).lower() == "spatial":
            if "spatial" in _adata.obsm.keys():
                coords = np.asarray(_adata.obsm["spatial"])
                coords = (
                    np.c_[coords, np.ones(shape=(coords.shape[0], 1))]
                    if coords.shape[1] == 2
                    else coords
                )
                active_actor.mapper.dataset.points = np.asarray(coords)
            else:
                print(f"!Warning: `spatial` is not included in anndata.obsm.")
        elif str(self._state[self.pcCOORDS]).lower() == "umap":
            if "X_umap" in _adata.obsm.keys():
                coords = np.asarray(_adata.obsm["X_umap"])
                coords = (
                    np.c_[coords, np.ones(shape=(coords.shape[0], 1))]
                    if coords.shape[1] == 2
                    else coords
                )
                active_actor.mapper.dataset.points = np.asarray(coords)
            else:
                print(f"!Warning: `X_umap` is not included in anndata.obsm.")
        else:
            pass

        self._plotter.actors[active_name] = active_actor
        self._plotter.view_isometric()
        self._ctrl.view_push_camera(force=True)
        self._ctrl.view_update()

    @vuwrap
    def on_opacity_change(self, **kwargs):
        _active_id = (
            1 if int(self._state.active_id) == 0 else int(self._state.active_id) - 1
        )
        active_name = self._state.actor_ids[_active_id]
        active_actor = self._plotter.actors[active_name]
        if str(active_name).startswith("PC"):
            active_actor.prop.opacity = float(self._state[self.pcOPACITY])
        else:
            active_actor.prop.opacity = float(self._state[self.meshOPACITY])
        self._ctrl.view_update()

    @vuwrap
    def on_ambient_change(self, **kwargs):
        _active_id = (
            1 if int(self._state.active_id) == 0 else int(self._state.active_id) - 1
        )
        active_name = self._state.actor_ids[_active_id]
        active_actor = self._plotter.actors[active_name]
        if str(active_name).startswith("PC"):
            active_actor.prop.ambient = float(self._state[self.pcAMBIENT])
        else:
            active_actor.prop.ambient = float(self._state[self.meshAMBIENT])
        self._ctrl.view_update()

    @vuwrap
    def on_color_change(self, **kwargs):
        if not self._state[self.pcCOLOR] in ["none", "None", None]:
            _active_id = (
                1 if int(self._state.active_id) == 0 else int(self._state.active_id) - 1
            )
            active_name = self._state.actor_ids[_active_id]
            active_actor = self._plotter.actors[active_name]
            if str(active_name).startswith("PC"):
                active_actor.prop.color = self._state[self.pcCOLOR]
                for morpho_key in ["MorphoField", "MorphoPath"]:
                    if morpho_key in self._plotter.actors.keys():
                        self._plotter.actors[morpho_key].prop.color = self._state[
                            self.pcCOLOR
                        ]
                self._ctrl.view_update()

        if not self._state[self.meshCOLOR] in ["none", "None", None]:
            _active_id = (
                1 if int(self._state.active_id) == 0 else int(self._state.active_id) - 1
            )
            active_name = self._state.actor_ids[_active_id]
            if str(active_name).startswith("Mesh"):
                active_actor = self._plotter.actors[active_name]
                active_actor.prop.color = self._state[self.meshCOLOR]
                self._ctrl.view_update()

    @vuwrap
    def on_colormap_change(self, **kwargs):
        _active_id = (
            1 if int(self._state.active_id) == 0 else int(self._state.active_id) - 1
        )
        active_name = self._state.actor_ids[_active_id]
        active_actor = self._plotter.actors[active_name]
        if str(active_name).startswith("PC"):
            active_actor.mapper.lookup_table.cmap = self._state[self.pcCOLORMAP]
            self.on_legend_change()
        for morpho_key in ["MorphoField", "MorphoPath"]:
            if morpho_key in self._plotter.actors.keys():
                self._plotter.actors[morpho_key].mapper.lookup_table.cmap = self._state[
                    self.pcCOLORMAP
                ]
        self._ctrl.view_update()

    @vuwrap
    def on_style_change(self, **kwargs):
        _active_id = (
            1 if int(self._state.active_id) == 0 else int(self._state.active_id) - 1
        )
        active_name = self._state.actor_ids[_active_id]
        active_actor = self._plotter.actors[active_name]
        if str(active_name).startswith("Mesh"):
            active_actor.prop.style = self._state[self.meshSTYLE]
        self._ctrl.view_update()

    @vuwrap
    def on_point_size_change(self, **kwargs):
        _active_id = (
            1 if int(self._state.active_id) == 0 else int(self._state.active_id) - 1
        )
        active_name = self._state.actor_ids[_active_id]
        active_actor = self._plotter.actors[active_name]
        if str(active_name).startswith("PC"):
            active_actor.prop.point_size = float(self._state[self.pcPOINTSIZE])
        self._ctrl.view_update()

    @vuwrap
    def on_morphology_change(self, **kwargs):
        if self._state[self.meshMORPHOLOGY]:
            _active_id = (
                1 if int(self._state.active_id) == 0 else int(self._state.active_id) - 1
            )
            active_name = self._state.actor_ids[_active_id]
            active_model = self._plotter.actors[active_name].mapper.dataset

            # Length, width and height of model
            model_bounds = np.asarray(active_model.bounds)
            model_x = round(abs(model_bounds[1] - model_bounds[0]), 5)
            model_y = round(abs(model_bounds[3] - model_bounds[2]), 5)
            model_z = round(abs(model_bounds[5] - model_bounds[4]), 5)

            # Surface area and Volume of model
            model_sa = round(active_model.area, 5)
            model_v = round(active_model.volume, 5)

            if "model_morphology" in self._plotter.actors.keys():
                self._plotter.remove_actor(self._plotter.actors["model_morphology"])
            self._plotter.add_text(
                text=f"Length (x) of model: {model_x}\n"
                f"Width (y) of model: {model_y}\n"
                f"Height (z) of model: {model_z}\n"
                f"Surface area of model: {model_sa}\n"
                f"Volume of model: {model_v}\n",
                font="arial",
                color="white",
                font_size=15,
                position="upper_left",
                name="model_morphology",
            )
        else:
            if "model_morphology" in self._plotter.actors.keys():
                self._plotter.remove_actor(self._plotter.actors["model_morphology"])
        self._ctrl.view_update()

    @vuwrap
    def on_cal_morphogenesis(self, **kwargs):
        if self._state[self.morphoCALCULATION]:
            if "MorphoField" in self._plotter.actors.keys():
                self._plotter.remove_actor(self._plotter.actors["MorphoField"])
            if "MorphoPath" in self._plotter.actors.keys():
                self._plotter.remove_actor(self._plotter.actors["MorphoPath"])

            # target anndata
            if self._state[self.morphoANNDATA] == "uploaded_target_anndata":
                if type(self._state[self.morphoUPLOADEDANNDATA]) is dict:
                    file = ClientFile(self._state[self.morphoUPLOADEDANNDATA])
                    if file.content:
                        with tempfile.NamedTemporaryFile(suffix=file.name) as path:
                            with open(path.name, "wb") as f:
                                f.write(file.content)
                            target_adata = abstract_anndata(path=path.name)
                else:
                    target_adata = abstract_anndata(
                        path=self._state[self.morphoUPLOADEDANNDATA]
                    )
            elif self._state[self.morphoANNDATA] is None:
                target_adata = None
            else:
                path = local_dataset_manager[self._state[self.morphoANNDATA]]
                target_adata = abstract_anndata(path=path)

            # source anndata
            _active_id = (
                1 if int(self._state.active_id) == 0 else int(self._state.active_id) - 1
            )
            active_name = self._state.actor_ids[_active_id]
            active_model = self._plotter.actors[active_name].mapper.dataset.copy()
            active_model_index = active_model.point_data["obs_index"]
            source_adata = abstract_anndata(path=self._state.anndata_path)[
                active_model_index, :
            ]

            # device
            try:
                import torch

                _device = str(self._state.morphoMAPPINGdevice).lower()
                _device = (
                    _device if _device != "cpu" and torch.cuda.is_available() else "cpu"
                )
            except:
                _device = "cpu"

            # Calculate morphogenesis
            from .pv_morphogenesis import morphogenesis

            pc_model, pc_vectors, trajectory_model, stages_X = morphogenesis(
                source_adata=source_adata,
                target_adata=target_adata,
                source_pc_model=active_model,
                mapping_device=_device,
                mapping_method=str(self._state[self.morphoMAPPINGmethod]),
                mapping_factor=float(self._state[self.morphoMAPPING]),
                morphofield_factor=int(self._state[self.morphoFIELD]),
                morphopath_t_end=int(self._state[self.morphoTEND]),
                morphopath_sampling=int(self._state[self.morphoSAMPLING]),
            )

            self._plotter.actors[active_name].mapper.dataset = pc_model
            self._state[self.morphoPREDICTEDMODELS] = stages_X
            morphofield_actor = self._plotter.add_mesh(
                pc_vectors,
                scalars="V_Z",
                style="surface",
                show_scalar_bar=False,
                name="MorphoField",
            )
            morphofield_actor.mapper.scalar_visibility = True
            morphofield_actor.SetVisibility(self._state[self.morphoSHOWFIELD])
            morphopath_actor = self._plotter.add_mesh(
                trajectory_model,
                scalars="V_Z",
                style="wireframe",
                line_width=3,
                show_scalar_bar=False,
                name="MorphoPath",
            )
            morphopath_actor.mapper.scalar_visibility = True
            morphopath_actor.SetVisibility(self._state[self.morphoSHOWTRAJECTORY])
            self._ctrl.view_update()

    @vuwrap
    def on_show_morpho_model_change(self, **kwargs):
        """Toggle morpho model visibility."""
        if "MorphoField" in self._plotter.actors.keys():
            morphofield_actor = self._plotter.actors["MorphoField"]
            morphofield_actor.SetVisibility(self._state[self.morphoSHOWFIELD])
        if "MorphoPath" in self._plotter.actors.keys():
            morphopath_actor = self._plotter.actors["MorphoPath"]
            morphopath_actor.SetVisibility(self._state[self.morphoSHOWTRAJECTORY])
        self._ctrl.view_update()

    @vuwrap
    def on_morphogenesis_animation(self, **kwargs):
        """Take morphogenesis animation."""
        if not (self._state[self.morphoANIMATION] in ["none", "None", None]):
            _filename = f"stv_image/{self._state[self.morphoANIMATION]}"
            Path("stv_image").mkdir(parents=True, exist_ok=True)
            if str(_filename).endswith(".mp4"):
                if self._state[self.morphoPREDICTEDMODELS] is not None:
                    _active_id = (
                        1
                        if int(self._state.active_id) == 0
                        else int(self._state.active_id) - 1
                    )
                    active_name = self._state.actor_ids[_active_id]
                    active_model = self._plotter.actors[
                        active_name
                    ].mapper.dataset.copy()
                    active_model_index = np.asarray(
                        active_model.point_data["obs_index"]
                    )

                    cells_index = np.asarray(self._state[self.morphoPREDICTEDMODELS][0])
                    cells_points = self._state[self.morphoPREDICTEDMODELS][1:]
                    if cells_index.shape == active_model_index.shape:
                        array = active_model.point_data[self._state[self.pcSCALARS]]
                        array = np.asarray(
                            pd.DataFrame(array, index=active_model_index).loc[
                                cells_index, 0
                            ]
                        )

                        cells_models = []
                        for pts in cells_points:
                            model = pv.PolyData(pts)
                            model.point_data[self._state[self.pcSCALARS]] = array
                            cells_models.append(model)

                        # Check models.
                        blocks = pv.MultiBlock(cells_models)
                        blocks_name = blocks.keys()

                        # Create another plotting object to save pyvista/vtk model.
                        pl = pv.Plotter(
                            window_size=(1024, 1024),
                            off_screen=True,
                            lighting="light_kit",
                        )
                        pl.background_color = "black"
                        pl.camera_position = self._plotter.camera_position
                        if (
                            self._state[self.morphoSHOWFIELD] is True
                            and "MorphoField" in self._plotter.actors.keys()
                        ):
                            morphofield_model = self._plotter.actors[
                                "MorphoField"
                            ].mapper.dataset.copy()
                            pl.add_mesh(
                                morphofield_model,
                                scalars=self._state[self.pcSCALARS],
                                style="surface",
                                ambient=0.2,
                                opacity=1.0,
                                cmap=self._state[self.pcCOLORMAP],
                            )
                        if (
                            self._state[self.morphoSHOWTRAJECTORY] is True
                            and "MorphoPath" in self._plotter.actors.keys()
                        ):
                            morphopath_model = self._plotter.actors[
                                "MorphoPath"
                            ].mapper.dataset.copy()
                            pl.add_mesh(
                                morphopath_model,
                                scalars=self._state[self.pcSCALARS],
                                style="wireframe",
                                line_width=3,
                                ambient=0.2,
                                opacity=1.0,
                                cmap=self._state[self.pcCOLORMAP],
                            )

                        start_block = blocks[blocks_name[0]].copy()
                        pl.add_mesh(
                            start_block,
                            scalars=self._state[self.pcSCALARS],
                            style="points",
                            point_size=5,
                            render_points_as_spheres=True,
                            ambient=0.2,
                            opacity=1.0,
                            cmap=self._state[self.pcCOLORMAP],
                        )
                        pl.open_movie(_filename, framerate=12, quality=5)
                        for block_name in blocks_name[1:]:
                            start_block.overwrite(blocks[block_name])
                            pl.write_frame()
                        pl.close()

    @vuwrap
    def on_plotter_screenshot(self, **kwargs):
        """Take screenshot."""
        if not (self._state[self.PLOTTER_SCREENSHOT] in ["none", "None", None]):
            _filename = f"stv_image/{self._state[self.PLOTTER_SCREENSHOT]}"
            Path("stv_image").mkdir(parents=True, exist_ok=True)
            if str(_filename).endswith(".png"):
                self._plotter.screenshot(filename=_filename)
            elif str(_filename).endswith(".pdf"):
                self._plotter.save_graphic(
                    filename=_filename,
                    title="PyVista Export",
                    raster=True,
                    painter=True,
                )

    @vuwrap
    def on_plotter_animation(self, **kwargs):
        """Take animation."""
        if not (self._state[self.PLOTTER_ANIMATION] in ["none", "None", None]):
            _filename = f"stv_image/{self._state[self.PLOTTER_ANIMATION]}"
            Path("stv_image").mkdir(parents=True, exist_ok=True)
            if str(_filename).endswith(".mp4"):
                viewup = self._plotter.camera_position[2]
                path = self._plotter.generate_orbital_path(
                    # factor=2.0,
                    # shift=0,
                    viewup=viewup,
                    n_points=int(self._state.animation_npoints),
                )
                self._plotter.open_movie(
                    _filename, framerate=int(self._state.animation_framerate), quality=5
                )
                self._plotter.orbit_on_path(
                    path, write_frames=True, viewup=viewup, step=0.1
                )

    ####################
    # Custom Callbacks #
    ####################

    @vuwrap
    def on_custom_callback(self, **kwargs):
        """RNA velocity."""
        if self._state["custom_func"] is True:
            if self._state["custom_analysis"] is True:
                if "custom_model" in self._plotter.actors.keys():
                    self._plotter.remove_actor(self._plotter.actors["custom_model"])
                _active_id = (
                    1
                    if int(self._state.active_id) == 0
                    else int(self._state.active_id) - 1
                )
                active_name = self._state.actor_ids[_active_id]
                active_model = self._plotter.actors[active_name].mapper.dataset.copy()
                active_model_index = active_model.point_data["obs_index"]
                adata = abstract_anndata(path=self._state.anndata_path)[
                    active_model_index, :
                ]

                # RNA velocity
                from .pv_custom import RNAvelocity

                pc_model, vectors = RNAvelocity(
                    adata=adata,
                    pc_model=active_model,
                    layer=str(self._state["custom_parameter1"]),
                    data_preprocess=str(self._state["custom_parameter2"]),
                    basis_pca=str(self._state["custom_parameter3"]),
                    basis_umap=str(self._state["custom_parameter4"]),
                    harmony_debatch=bool(self._state["custom_parameter5"]),
                    group_key=str(self._state["custom_parameter6"]),
                    n_neighbors=int(self._state["custom_parameter7"]),
                    n_pca_components=int(self._state["custom_parameter8"]),
                    n_vectors_downsampling=self._state["custom_parameter9"],
                    vectors_size=float(self._state["custom_parameter10"]),
                )

                self._plotter.actors[active_name].mapper.dataset = pc_model
                CustomModel_actor = self._plotter.add_mesh(
                    vectors,
                    scalars=f"speed_{str(self._state['custom_parameter3'])}",
                    style="surface",
                    show_scalar_bar=False,
                    name="custom_model",
                )
                CustomModel_actor.mapper.scalar_visibility = True
                CustomModel_actor.SetVisibility(self._state["custom_model_visible"])
                self._ctrl.view_update()

    @vuwrap
    def on_show_custom_model(self, **kwargs):
        """Toggle rna velocity vector model visibility."""
        if "custom_model" in self._plotter.actors.keys():
            custom_actor = self._plotter.actors["custom_model"]
            custom_actor.SetVisibility(self._state["custom_model_visible"])
        self._ctrl.view_update()
