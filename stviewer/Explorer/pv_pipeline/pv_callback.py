import gc
import os
import tempfile
from pathlib import Path

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import pyvista as pv
from anndata import AnnData
from scipy import sparse
from trame.app.file_upload import ClientFile

from stviewer.assets import local_dataset_manager

from .init_parameters import (
    init_adata_parameters,
    init_card_parameters,
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
        self.MEMORY_USAGE = f"{plotter._id_name}_memory_usage"

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
        self._state.change(self.MEMORY_USAGE)(self.on_memory_usage_change)

    @vuwrap
    def on_memory_usage_change(self, **kwargs):
        if self._state[self.MEMORY_USAGE]:
            if "model_memory" in self.plotter.actors.keys():
                self.plotter.remove_actor(self.plotter.actors["model_memory"])

            cbg_color = (
                "white" if self.plotter.background_color.name is "black" else "black"
            )
            self.plotter.add_text(
                text="Memory usage: %.4f GB"
                % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024),
                font="arial",
                color=cbg_color,
                font_size=10,
                position="upper_right",
                name="model_memory",
            )
        else:
            if "model_memory" in self.plotter.actors.keys():
                self.plotter.remove_actor(self.plotter.actors["model_memory"])
        self._ctrl.view_update()

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
    """Callbacks for Input based on pyvista."""

    def __init__(self, server, plotter):
        """Initialize SwitchModels."""
        state, ctrl = server.state, server.controller
        self._server = server
        self._ctrl = ctrl
        self._state = state
        self.plotter = plotter

        # State variable names
        self.SELECT_SAMPLES = "select_samples"
        self.UPLOAD_ANNDATA = "uploaded_anndata_path"

        # Listen to state changes
        self._state.change(self.SELECT_SAMPLES)(self.on_dataset_change)
        self._state.change(self.UPLOAD_ANNDATA)(self.on_anndata_change)

    @vuwrap
    def on_dataset_change(self, **kwargs):
        if self._state[self.SELECT_SAMPLES] is None:
            pass
        else:
            from stviewer.assets.dataset_acquisition import sample_dataset

            self._state[self.UPLOAD_ANNDATA] = None
            if self._state[self.SELECT_SAMPLES] == "uploaded_sample":
                path = self._state.selected_dir
            else:
                path = local_dataset_manager[self._state[self.SELECT_SAMPLES]]

            (
                anndata_info,
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
                    "anndata_info": anndata_info,
                    "available_obs": ["None"] + anndata_info["anndata_obs_keys"],
                    "available_genes": ["None"] + anndata_info["anndata_var_index"],
                    "pc_colormaps_list": ["spateo_cmap"]
                    + custom_colors
                    + plt.colormaps(),
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
                }
            )
            self._state.update(init_card_parameters)
            self._state.update(init_adata_parameters)
            self._state.update(init_pc_parameters)
            self._state.update(init_mesh_parameters)
            self._state.update(init_morphogenesis_parameters)
            self._state.update(init_output_parameters)
            self._ctrl.view_reset_camera(force=True)
            self._ctrl.view_update()

    @vuwrap
    def on_anndata_change(self, **kwargs):
        if self._state[self.UPLOAD_ANNDATA] is None:
            pass
        else:
            from stviewer.assets.dataset_acquisition import sample_dataset

            self._state[self.SELECT_SAMPLES] = None
            file = ClientFile(self._state[self.UPLOAD_ANNDATA])
            with tempfile.NamedTemporaryFile(suffix=file.name) as path:
                with open(path.name, "wb") as f:
                    f.write(file.content)
                    (
                        anndata_info,
                        pc_models,
                        pc_model_ids,
                        mesh_models,
                        mesh_model_ids,
                        custom_colors,
                    ) = sample_dataset(path=path.name)

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
                    "anndata_info": anndata_info,
                    "available_obs": ["None"] + anndata_info["anndata_obs_keys"],
                    "available_genes": ["None"] + anndata_info["anndata_var_index"],
                    "pc_colormaps_list": ["spateo_cmap"]
                    + custom_colors
                    + plt.colormaps(),
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
                }
            )
            self._state.update(init_card_parameters)
            self._state.update(init_adata_parameters)
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

        # Listen to state changes
        self._state.change("pc_obs_value")(self.on_obs_change)
        self._state.change("pc_gene_value")(self.on_gene_change)
        self._state.change("pc_matrix_value")(self.on_gene_change)
        self._state.change("pc_coords_value")(self.on_coords_change)
        self._state.change("pc_opacity_value")(self.on_opacity_change)
        self._state.change("pc_ambient_value")(self.on_ambient_change)
        self._state.change("pc_color_value")(self.on_color_change)
        self._state.change("pc_colormap_value")(self.on_colormap_change)
        self._state.change("pc_point_size_value")(self.on_point_size_change)
        self._state.change("pc_add_legend")(self.on_legend_change)
        self._state.change("pc_picking_group")(self.on_picking_pc_model)
        self._state.change("pc_overwrite")(self.on_picking_pc_model)
        self._state.change("pc_reload")(self.on_reload_main_model)

        self._state.change("mesh_opacity_value")(self.on_opacity_change)
        self._state.change("mesh_ambient_value")(self.on_ambient_change)
        self._state.change("mesh_color_value")(self.on_color_change)
        self._state.change("mesh_style_value")(self.on_style_change)
        self._state.change("mesh_morphology")(self.on_morphology_change)

        self._state.change("cal_morphogenesis")(self.on_cal_morphogenesis)
        self._state.change("morphofield_visibile")(self.on_show_morpho_model_change)
        self._state.change("morphopath_visibile")(self.on_show_morpho_model_change)
        self._state.change("morphopath_animation_path")(self.on_morphogenesis_animation)

        self._state.change("cal_interpolation")(self.on_cal_interpolation)

        self._state.change("screenshot_path")(self.on_plotter_screenshot)
        self._state.change("animation_path")(self.on_plotter_animation)

        # Custom controller
        if self._state.custom_func is True:
            self._state.change("custom_analysis")(self.on_custom_callback)
            self._state.change("custom_model_visible")(self.on_show_custom_model)

    @vuwrap
    def on_obs_change(self, **kwargs):
        _active_id = (
            1 if int(self._state.active_id) == 0 else int(self._state.active_id) - 1
        )
        active_name = self._state.actor_ids[_active_id]
        active_actor = self._plotter.actors[active_name]

        if str(active_name).startswith("PC"):
            if self._state.pc_obs_value in ["none", "None"]:
                active_actor.mapper.scalar_visibility = False
                for morpho_key in ["MorphoField", "MorphoPath"]:
                    if morpho_key in self._plotter.actors.keys():
                        self._plotter.actors[
                            morpho_key
                        ].mapper.scalar_visibility = False
                self.on_legend_change()
                self._ctrl.view_update()
            else:
                _obs_index = active_actor.mapper.dataset.point_data["obs_index"]
                if (
                    self._state.pc_obs_value
                    in self._state.anndata_info["anndata_obs_keys"]
                ):
                    self._state.pc_gene_value = None

                    change_array = True
                    array = active_actor.mapper.dataset[self._state.pc_obs_value].copy()

                    if np.issubdtype(array.dtype, np.number):
                        array = np.asarray(array, dtype=float)
                        self._state.pc_scalars_raw = {"None": "None"}
                    else:
                        od = {o: i for i, o in enumerate(np.unique(array))}
                        array = np.asarray(
                            list(map(lambda x: od[x], array)), dtype=float
                        )
                        self._state.pc_scalars_raw = od
                else:
                    array, change_array = None, False

                if change_array is True:
                    scalar_key = self._state.pc_obs_value
                    active_actor.mapper.dataset.point_data[f"{scalar_key}_vis"] = array
                    active_actor.mapper.scalar_range = (
                        active_actor.mapper.dataset.get_data_range(f"{scalar_key}_vis")
                    )

                    active_actor.mapper.SelectColorArray(f"{scalar_key}_vis")
                    active_actor.mapper.lookup_table.cmap = (
                        self._state.pc_colormap_value
                    )
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
                                f"{scalar_key}_vis"
                            ] = morpho_array
                            morpho_actor.mapper.scalar_range = (
                                active_actor.mapper.scalar_range
                            )
                            morpho_actor.mapper.SelectColorArray(f"{scalar_key}_vis")
                            morpho_actor.mapper.lookup_table.cmap = self._state[
                                f"{scalar_key}_vis"
                            ]
                            morpho_actor.mapper.SetScalarModeToUsePointFieldData()
                            morpho_actor.mapper.scalar_visibility = True
                            morpho_actor.mapper.Update()
                            self._plotter.actors[morpho_key] = morpho_actor
                else:
                    pass
                self._ctrl.view_update()

    @vuwrap
    def on_gene_change(self, **kwargs):
        _active_id = (
            1 if int(self._state.active_id) == 0 else int(self._state.active_id) - 1
        )
        active_name = self._state.actor_ids[_active_id]
        active_actor = self._plotter.actors[active_name]

        if str(active_name).startswith("PC"):
            if self._state.pc_gene_value in ["none", "None"]:
                active_actor.mapper.scalar_visibility = False
                for morpho_key in ["MorphoField", "MorphoPath"]:
                    if morpho_key in self._plotter.actors.keys():
                        self._plotter.actors[
                            morpho_key
                        ].mapper.scalar_visibility = False
                self.on_legend_change()
                self._ctrl.view_update()
            else:
                _obs_index = active_actor.mapper.dataset.point_data["obs_index"]
                if (
                    self._state.pc_gene_value
                    in self._state.anndata_info["anndata_var_index"]
                ):
                    self._state.pc_obs_value = None

                    change_array = True
                    obs_indices = np.where(
                        np.isin(
                            self._state.anndata_info["anndata_obs_index"], _obs_index
                        )
                    )[0]
                    var_indices = np.where(
                        np.isin(
                            self._state.anndata_info["anndata_var_index"],
                            [self._state.pc_gene_value],
                        )
                    )[0]

                    martix_path = f"{self._state.anndata_info['matrices_npz_path']}/{self._state.pc_matrix_value}_sparse_martrix.npz"
                    martix = sparse.load_npz(martix_path)
                    array = np.asarray(
                        martix[obs_indices, var_indices].A, dtype=float
                    ).reshape(-1, 1)
                    self._state.pc_scalars_raw = {"None": "None"}

                    del martix
                    gc.collect()
                elif (
                    self._state.pc_gene_value
                    in active_actor.mapper.dataset.point_data.keys()
                ):
                    self._state.pc_obs_value = None

                    change_array = True
                    array = np.asarray(
                        active_actor.mapper.dataset[
                            str(self._state.pc_gene_value)
                        ].copy(),
                        dtype=float,
                    )
                    self._state.pc_scalars_raw = {"None": "None"}
                else:
                    array, change_array = None, False

                if change_array is True:
                    scalar_key = self._state.pc_gene_value
                    active_actor.mapper.dataset.point_data[f"{scalar_key}_vis"] = array
                    active_actor.mapper.scalar_range = (
                        active_actor.mapper.dataset.get_data_range(f"{scalar_key}_vis")
                    )

                    active_actor.mapper.SelectColorArray(f"{scalar_key}_vis")
                    active_actor.mapper.lookup_table.cmap = (
                        self._state.pc_colormap_value
                    )
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
                                f"{scalar_key}_vis"
                            ] = morpho_array
                            morpho_actor.mapper.scalar_range = (
                                active_actor.mapper.scalar_range
                            )
                            morpho_actor.mapper.SelectColorArray(f"{scalar_key}_vis")
                            morpho_actor.mapper.lookup_table.cmap = self._state[
                                f"{scalar_key}_vis"
                            ]
                            morpho_actor.mapper.SetScalarModeToUsePointFieldData()
                            morpho_actor.mapper.scalar_visibility = True
                            morpho_actor.mapper.Update()
                            self._plotter.actors[morpho_key] = morpho_actor
                else:
                    pass
                self._ctrl.view_update()

    @vuwrap
    def on_legend_change(self, **kwargs):
        _active_id = (
            1 if int(self._state.active_id) == 0 else int(self._state.active_id) - 1
        )
        active_name = self._state.actor_ids[_active_id]
        active_actor = self._plotter.actors[active_name]

        if self._state.pc_add_legend and active_actor.mapper.scalar_visibility:
            if len(self._plotter.scalar_bars.keys()) != 0:
                self._plotter.remove_scalar_bar()
            if self._plotter.legend:
                self._plotter.remove_legend()

            cbg_color = (
                "white" if self._plotter.background_color.name is "black" else "black"
            )
            if "None" in self._state.pc_scalars_raw.keys():
                scalar_key = (
                    self._state.pc_obs_value
                    if self._state.pc_obs_value not in ["none", "None", None]
                    else self._state.pc_gene_value
                )

                self._plotter.add_scalar_bar(
                    scalar_key,
                    mapper=active_actor.mapper,
                    color=cbg_color,
                    bold=True,
                    interactive=False,
                    vertical=True,
                    title_font_size=25,
                    label_font_size=20,
                    outline=False,
                    fmt="%10.2f",
                )
            else:
                import matplotlib as mpl

                legend_labels = [i for i in self._state.pc_scalars_raw.keys()]

                lscmap = mpl.cm.get_cmap(self._state.pc_colormap_value)
                legend_hex = [
                    mpl.colors.to_hex(lscmap(i))
                    for i in np.linspace(0, 1, len(legend_labels))
                ]

                legend_entries = [
                    [label, hex] for label, hex in zip(legend_labels, legend_hex)
                ]
                legend_height = len(legend_entries) / 80
                legend_height = 0.05 if legend_height < 0.05 else legend_height
                legend_height = 1 if legend_height > 1 else legend_height
                self._plotter.add_legend(
                    legend_entries,
                    face="circle",
                    bcolor=None,
                    loc="center left",
                    size=(0.2, legend_height),
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
        if not (self._state.pc_picking_group in ["none", "None", None]):
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
                custom_picking_group = self._state.pc_picking_group
                added_active_model = active_model.extract_points(
                    active_model.point_data[self._state.pc_obs_value]
                    == float(custom_picking_group)
                )
            else:
                custom_picking_group = self._state.pc_scalars_raw[
                    self._state.pc_picking_group
                ]
                added_active_model = active_model.extract_points(
                    active_model.point_data[f"{self._state.pc_obs_value}_vis"]
                    == float(custom_picking_group)
                )
            if self._state.pc_overwrite is True:
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
            self._state.pc_picking_group = None
            self._state.pc_overwrite = False
        self._ctrl.view_update()

    @vuwrap
    def on_coords_change(self, **kwargs):
        _active_id = (
            1 if int(self._state.active_id) == 0 else int(self._state.active_id) - 1
        )
        active_name = self._state.actor_ids[_active_id]
        active_actor = self._plotter.actors[active_name]

        coords_key = str(self._state.pc_coords_value)
        if coords_key in self._state.anndata_info["anndata_obsm_keys"]:
            active_actor.mapper.dataset.points[
                :, 0
            ] = active_actor.mapper.dataset.point_data[f"{coords_key}_X"]
            active_actor.mapper.dataset.points[
                :, 1
            ] = active_actor.mapper.dataset.point_data[f"{coords_key}_Y"]
            active_actor.mapper.dataset.points[
                :, 2
            ] = active_actor.mapper.dataset.point_data[f"{coords_key}_Z"]
        else:
            print(f"!Warning: `{coords_key}` is not included in anndata.obsm.")

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
            active_actor.prop.opacity = float(self._state.pc_opacity_value)
        else:
            active_actor.prop.opacity = float(self._state.mesh_opacity_value)
        self._ctrl.view_update()

    @vuwrap
    def on_ambient_change(self, **kwargs):
        _active_id = (
            1 if int(self._state.active_id) == 0 else int(self._state.active_id) - 1
        )
        active_name = self._state.actor_ids[_active_id]
        active_actor = self._plotter.actors[active_name]
        if str(active_name).startswith("PC"):
            active_actor.prop.ambient = float(self._state.pc_ambient_value)
        else:
            active_actor.prop.ambient = float(self._state.mesh_ambient_value)
        self._ctrl.view_update()

    @vuwrap
    def on_color_change(self, **kwargs):
        if not self._state.pc_color_value in ["none", "None", None]:
            _active_id = (
                1 if int(self._state.active_id) == 0 else int(self._state.active_id) - 1
            )
            active_name = self._state.actor_ids[_active_id]
            active_actor = self._plotter.actors[active_name]
            if str(active_name).startswith("PC"):
                active_actor.prop.color = self._state.pc_color_value
                for morpho_key in ["MorphoField", "MorphoPath"]:
                    if morpho_key in self._plotter.actors.keys():
                        self._plotter.actors[
                            morpho_key
                        ].prop.color = self._state.pc_color_value
                self._ctrl.view_update()

        if not self._state.pc_color_value in ["none", "None", None]:
            _active_id = (
                1 if int(self._state.active_id) == 0 else int(self._state.active_id) - 1
            )
            active_name = self._state.actor_ids[_active_id]
            if str(active_name).startswith("Mesh"):
                active_actor = self._plotter.actors[active_name]
                active_actor.prop.color = self._state.pc_color_value
                self._ctrl.view_update()

    @vuwrap
    def on_colormap_change(self, **kwargs):
        _active_id = (
            1 if int(self._state.active_id) == 0 else int(self._state.active_id) - 1
        )
        active_name = self._state.actor_ids[_active_id]
        active_actor = self._plotter.actors[active_name]
        if str(active_name).startswith("PC"):
            active_actor.mapper.lookup_table.cmap = self._state.pc_colormap_value
            self.on_legend_change()
        for morpho_key in ["MorphoField", "MorphoPath"]:
            if morpho_key in self._plotter.actors.keys():
                self._plotter.actors[
                    morpho_key
                ].mapper.lookup_table.cmap = self._state.pc_colormap_value
        self._ctrl.view_update()

    @vuwrap
    def on_style_change(self, **kwargs):
        _active_id = (
            1 if int(self._state.active_id) == 0 else int(self._state.active_id) - 1
        )
        active_name = self._state.actor_ids[_active_id]
        active_actor = self._plotter.actors[active_name]
        if str(active_name).startswith("Mesh"):
            active_actor.prop.style = self._state.mesh_style_value
        self._ctrl.view_update()

    @vuwrap
    def on_point_size_change(self, **kwargs):
        _active_id = (
            1 if int(self._state.active_id) == 0 else int(self._state.active_id) - 1
        )
        active_name = self._state.actor_ids[_active_id]
        active_actor = self._plotter.actors[active_name]
        if str(active_name).startswith("PC"):
            active_actor.prop.point_size = float(self._state.pc_point_size_value)
        self._ctrl.view_update()

    @vuwrap
    def on_morphology_change(self, **kwargs):
        if self._state.mesh_morphology:
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

            cbg_color = (
                "white" if self._plotter.background_color.name is "black" else "black"
            )
            self._plotter.add_text(
                text=f"Length (x) of model: {model_x}\n"
                f"Width (y) of model: {model_y}\n"
                f"Height (z) of model: {model_z}\n"
                f"Surface area of model: {model_sa}\n"
                f"Volume of model: {model_v}\n",
                font="arial",
                color=cbg_color,
                font_size=10,
                position="lower_left",
                name="model_morphology",
            )
        else:
            if "model_morphology" in self._plotter.actors.keys():
                self._plotter.remove_actor(self._plotter.actors["model_morphology"])
        self._ctrl.view_update()

    @vuwrap
    def on_cal_morphogenesis(self, **kwargs):
        if self._state.cal_morphogenesis:
            from stviewer.assets.dataset_acquisition import abstract_anndata

            if "MorphoField" in self._plotter.actors.keys():
                self._plotter.remove_actor(self._plotter.actors["MorphoField"])
            if "MorphoPath" in self._plotter.actors.keys():
                self._plotter.remove_actor(self._plotter.actors["MorphoPath"])

            # target anndata
            if self._state.morpho_target_anndata_path == "uploaded_target_anndata":
                if type(self._state.morpho_uploaded_target_anndata_path) is dict:
                    file = ClientFile(self._state.morpho_uploaded_target_anndata_path)
                    if file.content:
                        with tempfile.NamedTemporaryFile(suffix=file.name) as path:
                            with open(path.name, "wb") as f:
                                f.write(file.content)
                            target_adata, _ = abstract_anndata(path=path.name)
                else:
                    target_adata, _ = abstract_anndata(
                        path=self._state.morpho_uploaded_target_anndata_path
                    )
            elif self._state.morpho_target_anndata_path is None:
                target_adata = None
            else:
                path = local_dataset_manager[self._state.morpho_target_anndata_path]
                target_adata, _ = abstract_anndata(path=path)

            # source anndata
            _active_id = (
                1 if int(self._state.active_id) == 0 else int(self._state.active_id) - 1
            )
            active_name = self._state.actor_ids[_active_id]
            active_model = self._plotter.actors[active_name].mapper.dataset.copy()
            active_model_index = active_model.point_data["obs_index"]
            source_adata, _ = abstract_anndata(
                path=self._state.anndata_info["anndata_path"]
            )[active_model_index, :]

            # device
            try:
                import torch

                _device = str(self._state.morpho_mapping_device).lower()
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
                mapping_method=str(self._state.morpho_mapping_method),
                mapping_factor=float(self._state.morpho_mapping_factor),
                morphofield_factor=int(self._state.morphofield_factor),
                morphopath_t_end=int(self._state.morphopath_t_end),
                morphopath_sampling=int(self._state.morphopath_downsampling),
            )

            self._plotter.actors[active_name].mapper.dataset = pc_model
            self._state.morphopath_predicted_models = stages_X
            morphofield_actor = self._plotter.add_mesh(
                pc_vectors,
                scalars="V_Z",
                style="surface",
                show_scalar_bar=False,
                name="MorphoField",
            )
            morphofield_actor.mapper.scalar_visibility = True
            morphofield_actor.SetVisibility(self._state.morphofield_visibile)
            morphopath_actor = self._plotter.add_mesh(
                trajectory_model,
                scalars="V_Z",
                style="wireframe",
                line_width=3,
                show_scalar_bar=False,
                name="MorphoPath",
            )
            morphopath_actor.mapper.scalar_visibility = True
            morphopath_actor.SetVisibility(self._state.morphopath_visibile)
            self._ctrl.view_update()

    @vuwrap
    def on_show_morpho_model_change(self, **kwargs):
        """Toggle morpho model visibility."""
        if "MorphoField" in self._plotter.actors.keys():
            morphofield_actor = self._plotter.actors["MorphoField"]
            morphofield_actor.SetVisibility(self._state.morphofield_visibile)
        if "MorphoPath" in self._plotter.actors.keys():
            morphopath_actor = self._plotter.actors["MorphoPath"]
            morphopath_actor.SetVisibility(self._state.morphopath_visibile)
        self._ctrl.view_update()

    @vuwrap
    def on_morphogenesis_animation(self, **kwargs):
        """Take morphogenesis animation."""
        if not (self._state.morphopath_animation_path in ["none", "None", None]):
            _filename = f"stv_image/{self._state.morphopath_animation_path}"
            Path("stv_image").mkdir(parents=True, exist_ok=True)
            if str(_filename).endswith(".mp4"):
                if self._state.morphopath_predicted_models is not None:
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

                    cells_index = np.asarray(self._state.morphopath_predicted_models[0])
                    cells_points = self._state.morphopath_predicted_models[1:]
                    if cells_index.shape == active_model_index.shape:
                        scalar_key = (
                            self._state.pc_obs_value
                            if self._state.pc_obs_value not in ["none", "None", None]
                            else self._state.pc_gene_value
                        )
                        array = active_model.point_data[f"{scalar_key}_vis"]
                        array = np.asarray(
                            pd.DataFrame(array, index=active_model_index).loc[
                                cells_index, 0
                            ]
                        )

                        cells_models = []
                        for pts in cells_points:
                            model = pv.PolyData(pts)
                            model.point_data[f"{scalar_key}_vis"] = array
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
                            self._state.morphofield_visibile is True
                            and "MorphoField" in self._plotter.actors.keys()
                        ):
                            morphofield_model = self._plotter.actors[
                                "MorphoField"
                            ].mapper.dataset.copy()
                            pl.add_mesh(
                                morphofield_model,
                                scalars=f"{scalar_key}_vis",
                                style="surface",
                                ambient=0.2,
                                opacity=1.0,
                                cmap=self._state.pc_colormap_value,
                            )
                        if (
                            self._state.morphopath_visibile is True
                            and "MorphoPath" in self._plotter.actors.keys()
                        ):
                            morphopath_model = self._plotter.actors[
                                "MorphoPath"
                            ].mapper.dataset.copy()
                            pl.add_mesh(
                                morphopath_model,
                                scalars=f"{scalar_key}_vis",
                                style="wireframe",
                                line_width=3,
                                ambient=0.2,
                                opacity=1.0,
                                cmap=self._state.pc_colormap_value,
                            )

                        start_block = blocks[blocks_name[0]].copy()
                        pl.add_mesh(
                            start_block,
                            scalars=f"{scalar_key}_vis",
                            style="points",
                            point_size=5,
                            render_points_as_spheres=True,
                            ambient=0.2,
                            opacity=1.0,
                            cmap=self._state.pc_colormap_value,
                        )
                        pl.open_movie(_filename, framerate=12, quality=5)
                        for block_name in blocks_name[1:]:
                            start_block.overwrite(blocks[block_name])
                            pl.write_frame()
                        pl.close()

    @vuwrap
    def on_cal_interpolation(self, **kwargs):
        """Learn a continuous mapping from space to gene expression pattern with the Gaussian Process method."""
        if self._state.cal_interpolation and self._state.pc_gene_value not in [
            "none",
            "None",
            None,
        ]:
            # source anndata
            _active_id = (
                1 if int(self._state.active_id) == 0 else int(self._state.active_id) - 1
            )
            active_name = self._state.actor_ids[_active_id]
            active_actor = self._plotter.actors[active_name]
            gene = self._state.pc_gene_value

            source_adata = AnnData(
                X=np.asarray(
                    active_actor.mapper.dataset.point_data[f"{gene}_vis"]
                ).reshape(-1, 1),
                obs=pd.DataFrame(
                    index=active_actor.mapper.dataset.point_data["obs_index"]
                ),
                var=pd.DataFrame(index=[gene]),
            )
            source_adata.obsm["target_points"] = np.asarray(
                active_actor.mapper.dataset.points
            )

            # device
            try:
                import torch

                _device = str(self._state.interpolation_device).lower()
                _device = (
                    _device if _device != "cpu" and torch.cuda.is_available() else "cpu"
                )
            except:
                _device = "cpu"

            # Calculate interpolated gene pattern
            from .pv_interpolation import gp_interpolation

            interpolated_adata = gp_interpolation(
                source_adata=source_adata,
                target_points=np.asarray(source_adata.obsm["target_points"]),
                spatial_key="target_points",
                keys=gene,
                device=_device,
                training_iter=100,
            )
            active_actor.mapper.dataset.point_data[
                f"{self._state.pc_gene_value}_interpolated"
            ] = np.asarray(interpolated_adata[:, gene].X.flatten())
            self._plotter.actors[active_name] = active_actor
            self._ctrl.view_update()

            del source_adata
            gc.collect()

    @vuwrap
    def on_plotter_screenshot(self, **kwargs):
        """Take screenshot."""
        if not (self._state.screenshot_path in ["none", "None", None]):
            _filename = f"stv_image/{self._state.screenshot_path}"
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
        if not (self._state.animation_path in ["none", "None", None]):
            _filename = f"stv_image/{self._state.animation_path}"
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
                from stviewer.assets.dataset_acquisition import abstract_anndata

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
                adata, _ = abstract_anndata(
                    path=self._state.anndata_info["anndata_path"]
                )
                adata = adata[active_model_index, :]

                # RNA velocity
                from .pv_custom import RNAvelocity

                pc_model, vectors = RNAvelocity(
                    adata=adata,
                    pc_model=active_model,
                    layer=self._state.custom_parameter1,
                    data_preprocess=self._state.custom_parameter2,
                    basis_pca=self._state.custom_parameter3,
                    basis_umap=self._state.custom_parameter4,
                    harmony_debatch=bool(self._state.custom_parameter5),
                    group_key=self._state.custom_parameter6,
                    n_neighbors=int(self._state.custom_parameter7),
                    n_pca_components=int(self._state.custom_parameter8),
                    n_vectors_downsampling=self._state.custom_parameter9,
                    vectors_size=float(self._state.custom_parameter10),
                )

                self._plotter.actors[active_name].mapper.dataset = pc_model
                CustomModel_actor = self._plotter.add_mesh(
                    vectors,
                    scalars=f"speed_{str(self._state.custom_parameter3)}",
                    style="surface",
                    show_scalar_bar=False,
                    name="custom_model",
                )
                CustomModel_actor.mapper.scalar_visibility = True
                CustomModel_actor.SetVisibility(self._state.custom_model_visible)
                self._ctrl.view_update()

    @vuwrap
    def on_show_custom_model(self, **kwargs):
        """Toggle rna velocity vector model visibility."""
        if "custom_model" in self._plotter.actors.keys():
            custom_actor = self._plotter.actors["custom_model"]
            custom_actor.SetVisibility(self._state.custom_model_visible)
        self._ctrl.view_update()
