import os
import warnings

warnings.filterwarnings("ignore")

from .pv_plotter import add_single_model

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from typing import Optional

from stviewer.assets import sample_dataset


def generate_actors(
    plotter,
    pc_models: Optional[list] = None,
    mesh_models: Optional[list] = None,
    pc_model_names: Optional[list] = None,
    mesh_model_names: Optional[list] = None,
):
    # Generate actors for pc models
    pc_kwargs = dict(model_style="points", model_size=8)
    if not (pc_models is None):
        pc_actors = [
            add_single_model(
                plotter=plotter, model=model, model_name=model_name, **pc_kwargs
            )
            for model, model_name in zip(pc_models, pc_model_names)
        ]
    else:
        pc_actors = None

    # Generate actors for mesh models
    mesh_kwargs = dict(opacity=0.6, model_style="surface")
    if not (mesh_models is None):
        mesh_actors = [
            add_single_model(
                plotter=plotter, model=model, model_name=model_name, **mesh_kwargs
            )
            for model, model_name in zip(mesh_models, mesh_model_names)
        ]
    else:
        mesh_actors = None
    return pc_actors, mesh_actors


def standard_tree(actors: list, base_id: int = 0):
    actor_tree, actor_names = [], []
    for i, actor in enumerate(actors):
        if i == 0:
            actor.SetVisibility(True)
        else:
            actor.SetVisibility(False)
        actor_names.append(str(actor.name))
        actor_tree.append(
            {
                "id": str(base_id + 1 + i),
                "parent": str(0) if i == 0 else str(base_id + 1),
                "visible": True if i == 0 else False,
                "name": str(actor.name),
            }
        )

    return actors, actor_names, actor_tree


def generate_actors_tree(
    pc_actors: Optional[list] = None,
    mesh_actors: Optional[list] = None,
):
    if not (pc_actors is None):
        pc_actors, pc_actor_names, pc_tree = standard_tree(actors=pc_actors, base_id=0)
    else:
        pc_actors, pc_actor_names, pc_tree = [], [], []

    if not (mesh_actors is None):
        mesh_actors, mesh_actor_names, mesh_tree = standard_tree(
            actors=mesh_actors,
            base_id=0 if pc_actors is None else len(pc_actors),
        )
    else:
        mesh_actors, mesh_actor_names, mesh_tree = [], [], []

    actors = pc_actors + mesh_actors
    actor_names = pc_actor_names + mesh_actor_names
    actor_tree = pc_tree + mesh_tree
    return actors, actor_names, actor_tree


def init_actors(plotter, path):
    (
        anndata_info,
        pc_models,
        pc_model_ids,
        mesh_models,
        mesh_model_ids,
        custom_colors,
    ) = sample_dataset(path=path)

    # Generate actors
    pc_actors, mesh_actors = generate_actors(
        plotter=plotter,
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

    return (
        anndata_info,
        actors,
        actor_names,
        actor_tree,
        custom_colors,
    )
