import io
import os
from pathlib import Path

import matplotlib.colors as mc
import numpy as np
import pyvista as pv

from stviewer.assets import local_dataset_manager
from stviewer.assets.dataset_acquisition import abstract_anndata, sample_dataset

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

        # Listen to state changes
        self._state.change(self.SELECT_SAMPLES)(self.on_dataset_change)

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
                mm_models,
                mm_model_ids,
            ) = sample_dataset(path=path)

            # Generate actors
            self.plotter.clear_actors()
            pc_actors, mesh_actors, mm_actors = generate_actors(
                plotter=self.plotter,
                pc_models=pc_models,
                mesh_models=mesh_models,
                mm_models=mm_models,
                mm_model_ids=mm_model_ids,
            )

            # Generate the relationship tree of actors
            actors, actor_ids, actor_tree = generate_actors_tree(
                pc_actors=pc_actors,
                pc_actor_ids=pc_model_ids,
                mesh_actors=mesh_actors,
                mesh_actor_ids=mesh_model_ids,
            )

            self._state.init_dataset = False
            self._state.sample_adata_path = os.path.join(
                os.path.join(path, "h5ad"),
                os.listdir(path=os.path.join(path, "h5ad"))[0],
            )
            self._state.actor_ids = actor_ids
            self._state.mm_actor_ids = mm_model_ids
            self._state.pipeline = actor_tree
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
        self.SCALARS = f"actor_scalars_value"
        self.MATRIX = f"actor_matrix_value"
        self.OPACITY = f"actor_opacity_value"
        self.AMBIENT = f"actor_ambient_value"
        self.COLOR = f"actor_color_value"
        self.COLORMAP = f"actor_colormap_value"
        self.STYLE = f"actor_style_value"
        self.POINTSIZE = f"actor_point_size_value"
        self.LINEWIDTH = f"actor_line_width_value"
        self.ASSPHERES = f"actor_as_spheres_value"
        self.ASTUBES = f"actor_as_tubes_value"
        self.SHOW_VECTORPC = f"show_vectorpc"
        self.SHOW_VECTORMESH = f"show_vectormesh"
        self.SHOW_TRAJECTORY = f"show_trajectory"
        self.MM_COLOR = f"mm_actor_color_value"
        self.MM_COLORMAP = f"mm_actor_colormap_value"
        self.PLOTTER_SCREENSHOT = "screenshot_path"
        self.PLOTTER_ANIMATION = "animation_path"

        # Listen to state changes
        self._state.change(self.SCALARS)(self.on_scalars_change)
        self._state.change(self.MATRIX)(self.on_scalars_change)
        self._state.change(self.OPACITY)(self.on_opacity_change)
        self._state.change(self.AMBIENT)(self.on_ambient_change)
        self._state.change(self.COLOR)(self.on_color_change)
        self._state.change(self.COLORMAP)(self.on_colormap_change)
        self._state.change(self.STYLE)(self.on_style_change)
        self._state.change(self.POINTSIZE)(self.on_point_size_change)
        self._state.change(self.LINEWIDTH)(self.on_line_width_change)
        self._state.change(self.ASSPHERES)(self.on_as_spheres_change)
        self._state.change(self.ASTUBES)(self.on_as_tubes_change)
        self._state.change(self.SHOW_VECTORPC)(self.on_show_vectorpc_change)
        self._state.change(self.SHOW_VECTORMESH)(self.on_show_vectormesh_change)
        self._state.change(self.SHOW_TRAJECTORY)(self.on_show_trajectory_change)
        self._state.change(self.MM_COLOR)(self.on_mm_color_change)
        self._state.change(self.MM_COLORMAP)(self.on_mm_colormap_change)
        self._state.change(self.PLOTTER_SCREENSHOT)(self.on_plotter_screenshot)
        self._state.change(self.PLOTTER_ANIMATION)(self.on_plotter_animation)

    @vuwrap
    def on_plotter_screenshot(self, **kwargs):
        """Take screenshot."""
        if not (self._state[self.PLOTTER_SCREENSHOT] in ["none", "None", None]):
            _filename = f"stv_image/{self._state[self.PLOTTER_SCREENSHOT]}"
            Path("stv_image").mkdir(parents=True, exist_ok=True)
            if str(_filename).endswith("png"):
                self._plotter.screenshot(filename=_filename)
            elif str(_filename).endswith("pdf"):
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
            if str(_filename).endswith("mp4"):
                viewup = self._plotter.camera.GetViewUp()
                view_x, view_y, view_z = self._plotter.camera.GetViewPlaneNormal()
                path = self._plotter.generate_orbital_path(
                    factor=2.0,
                    shift=0,
                    viewup=None,
                    n_points=int(self._state.animation_npoints),
                )
                self._plotter.open_movie(
                    _filename, framerate=int(self._state.animation_framerate), quality=5
                )
                self._plotter.orbit_on_path(
                    path, write_frames=True, step=0.1
                )  # viewup=(0, 0, 1)
                # self._plotter.close()

    @vuwrap
    def on_scalars_change(self, **kwargs):
        active_actor = [value for value in self._plotter.actors.values()][
            int(self._state.active_id) - 1
        ]
        active_mm_actor = (
            None
            if self._state.active_mm_id is None
            else [value for value in self._plotter.actors.values()][
                len(self._state.actor_ids) + self._state.active_mm_id
            ]
        )
        self._actor = active_actor

        if self._state[self.SCALARS] in ["none", "None", None]:
            active_actor.mapper.scalar_visibility = False
            if not (active_mm_actor is None):
                active_mm_actor.mapper.scalar_visibility = False
        else:
            _raw_adata = abstract_anndata(path=self._state.sample_adata_path)
            for i, _active_actor in enumerate([active_actor, active_mm_actor]):
                if not (_active_actor is None):
                    _obs_index = _active_actor.mapper.dataset.point_data["obs_index"]
                    _adata = _raw_adata[_obs_index, :].copy()
                    if self._state[self.SCALARS] in set(_adata.obs_keys()):
                        array = _adata.obs[self._state[self.SCALARS]].values
                        if array.dtype == "category":
                            array = np.asarray(array, dtype=str)
                        if np.issubdtype(array.dtype, np.number):
                            array = np.asarray(array, dtype=float)
                        else:
                            od = {o: i for i, o in enumerate(np.unique(array))}
                            array = np.asarray(
                                list(map(lambda x: od[x], array)), dtype=float
                            )
                        array = array.reshape(-1, 1)
                    elif self._state[self.SCALARS] in set(_adata.var_names.tolist()):
                        matrix_id = self._state[self.MATRIX]
                        if matrix_id == "X_counts":
                            array = np.asarray(
                                _adata[:, self._state[self.SCALARS]].X.sum(axis=1),
                                dtype=float,
                            )
                        else:
                            array = np.asarray(
                                _adata[:, self._state[self.SCALARS]]
                                .layers[matrix_id]
                                .sum(axis=1),
                                dtype=float,
                            )
                    else:
                        array = np.ones(shape=(len(_obs_index), 1), dtype=float)

                    _active_actor.mapper.dataset.point_data[
                        self._state[self.SCALARS]
                    ] = array
                    _active_actor.mapper.scalar_range = (
                        _active_actor.mapper.dataset.get_data_range(
                            self._state[self.SCALARS]
                        )
                    )

                    _active_actor.mapper.SelectColorArray(self._state[self.SCALARS])
                    _active_actor.mapper.lookup_table.cmap = self._state[self.COLORMAP]
                    _active_actor.mapper.SetScalarModeToUsePointFieldData()
                    _active_actor.mapper.scalar_visibility = True
        self._ctrl.view_update()

    def on_opacity_change(self, **kwargs):
        active_actor = [value for value in self._plotter.actors.values()][
            int(self._state.active_id) - 1
        ]
        active_actor.prop.opacity = self._state[self.OPACITY]
        self._ctrl.view_update()

    def on_ambient_change(self, **kwargs):
        active_actor = [value for value in self._plotter.actors.values()][
            int(self._state.active_id) - 1
        ]
        active_actor.prop.ambient = self._state[self.AMBIENT]
        self._ctrl.view_update()

    def on_color_change(self, **kwargs):
        active_actor = [value for value in self._plotter.actors.values()][
            int(self._state.active_id) - 1
        ]
        active_actor.prop.color = self._state[self.COLOR]
        self._ctrl.view_update()

    def on_colormap_change(self, **kwargs):
        active_actor = [value for value in self._plotter.actors.values()][
            int(self._state.active_id) - 1
        ]
        active_actor.mapper.lookup_table.cmap = self._state[self.COLORMAP]
        self._ctrl.view_update()

    def on_style_change(self, **kwargs):
        active_actor = [value for value in self._plotter.actors.values()][
            int(self._state.active_id) - 1
        ]
        active_actor.prop.style = self._state[self.STYLE]
        self._ctrl.view_update()

    def on_point_size_change(self, **kwargs):
        active_actor = [value for value in self._plotter.actors.values()][
            int(self._state.active_id) - 1
        ]
        active_actor.prop.point_size = self._state[self.POINTSIZE]
        self._ctrl.view_update()

    def on_line_width_change(self, **kwargs):
        active_actor = [value for value in self._plotter.actors.values()][
            int(self._state.active_id) - 1
        ]
        active_actor.prop.line_width = self._state[self.LINEWIDTH]
        self._ctrl.view_update()

    def on_as_spheres_change(self, **kwargs):
        active_actor = [value for value in self._plotter.actors.values()][
            int(self._state.active_id) - 1
        ]
        active_actor.prop.render_points_as_spheres = self._state[self.ASSPHERES]
        self._ctrl.view_update()

    def on_as_tubes_change(self, **kwargs):
        active_actor = [value for value in self._plotter.actors.values()][
            int(self._state.active_id) - 1
        ]
        active_actor.prop.render_lines_as_tubes = self._state[self.ASTUBES]
        self._ctrl.view_update()

    def on_show_vectorpc_change(self, **kwargs):
        if not (self._state.mm_actor_ids is None):
            _title = self._state.actor_ids[self._state.active_id - 1].split("_")[1]
            _mm_ids = list(self._state.mm_actor_ids)
            if True in list(map(lambda x: str(x).startswith(_title), _mm_ids)):
                mm_id_index = _mm_ids.index(f"{_title}_VectorPC")
                active_vectorpc = [value for value in self._plotter.actors.values()][
                    len(self._state.actor_ids) + mm_id_index
                ]
                active_vectorpc.SetVisibility(self._state[self.SHOW_VECTORPC])
                self._state.active_mm_id = (
                    mm_id_index if self._state[self.SHOW_VECTORPC] == True else None
                )
                self._ctrl.view_update()

    def on_show_vectormesh_change(self, **kwargs):
        if not (self._state.mm_actor_ids is None):
            _title = self._state.actor_ids[self._state.active_id - 1].split("_")[1]
            _mm_ids = list(self._state.mm_actor_ids)
            if True in list(map(lambda x: str(x).startswith(_title), _mm_ids)):
                mm_id_index = _mm_ids.index(f"{_title}_VectorMesh")
                active_vectormesh = [value for value in self._plotter.actors.values()][
                    len(self._state.actor_ids) + mm_id_index
                ]
                active_vectormesh.SetVisibility(self._state[self.SHOW_VECTORMESH])
                self._state.active_mm_id = (
                    mm_id_index if self._state[self.SHOW_VECTORMESH] == True else None
                )
                self._ctrl.view_update()

    def on_show_trajectory_change(self, **kwargs):
        if not (self._state.mm_actor_ids is None):
            _title = self._state.actor_ids[self._state.active_id - 1].split("_")[1]
            _mm_ids = list(self._state.mm_actor_ids)
            if True in list(map(lambda x: str(x).startswith(_title), _mm_ids)):
                mm_id_index = _mm_ids.index(f"{_title}_Trajectory")
                active_trajectory = [value for value in self._plotter.actors.values()][
                    len(self._state.actor_ids) + mm_id_index
                ]
                active_trajectory.SetVisibility(self._state[self.SHOW_TRAJECTORY])
                self._state.active_mm_id = (
                    mm_id_index if self._state[self.SHOW_TRAJECTORY] == True else None
                )
                self._ctrl.view_update()

    def on_mm_color_change(self, **kwargs):
        if not (self._state.active_mm_id is None):
            active_mm_actor = [value for value in self._plotter.actors.values()][
                len(self._state.actor_ids) + self._state.active_mm_id
            ]
            active_mm_actor.prop.color = self._state[self.MM_COLOR]
            self._ctrl.view_update()

    def on_mm_colormap_change(self, **kwargs):
        if not (self._state.active_mm_id is None):
            active_mm_actor = [value for value in self._plotter.actors.values()][
                len(self._state.actor_ids) + self._state.active_mm_id
            ]
            active_mm_actor.mapper.lookup_table.cmap = self._state[self.MM_COLORMAP]
            self._ctrl.view_update()
