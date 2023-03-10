
from .pv_plotter import add_single_model, create_plotter

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from typing import Optional


def generate_actors(
    pc_models: Optional[list] = None,
    mesh_models: Optional[list] = None,
    pc_added_kwargs: Optional[dict] = None,
    mesh_added_kwargs: Optional[dict] = None,
    **kwargs
):
    # Generate a new plotter
    plotter = create_plotter(**kwargs)
    # Generate actors for pc models
    if not (pc_models is None):
        if pc_added_kwargs is None:
            pc_actors = [add_single_model(plotter=plotter, model=model) for model in pc_models]
        else:
            pc_actors = [add_single_model(plotter=plotter, model=model, **pc_added_kwargs) for model in pc_models]
    else:
        pc_actors = None
    # Generate actors for mesh models
    if not (mesh_models is None):
        if mesh_added_kwargs is None:
            mesh_actors = [add_single_model(plotter=plotter, model=model) for model in mesh_models]
        else:
            mesh_actors = [add_single_model(plotter=plotter, model=model, **mesh_added_kwargs) for model in mesh_models]
    else:
        mesh_actors = None
    return plotter, pc_actors, mesh_actors


def standard_tree(actors: list, actor_names: list, base_id: int = 0):
    for i, actor in enumerate(actors):
        if i == 0:
            actor.SetVisibility(True)
        else:
            actor.SetVisibility(False)
    tree = [
        {
            "id": str(base_id + 1 + i),
            "parent": str(0) if i == 0 else str(base_id + 1),
            "visible": True if i == 0 else False,
            "name": actor_names[i],
        }
        for i, name in enumerate(actor_names)
    ]
    return actors, actor_names, tree


def generate_actors_tree(
    pc_actors: Optional[list] = None,
    pc_actor_ids: Optional[list] = None,
    mesh_actors: Optional[list] = None,
    mesh_actor_ids: Optional[list] = None,
):
    if not (pc_actors is None):
        pc_actors, pc_actor_ids, pc_tree = standard_tree(
            actors=pc_actors, actor_names=pc_actor_ids, base_id=0
        )
    else:
        pc_actors, pc_actor_ids, pc_tree = None, None, None

    if not (mesh_actors is None):
        mesh_actors, mesh_actor_ids, mesh_tree = standard_tree(
            actors=mesh_actors, actor_names=mesh_actor_ids, base_id=0 if pc_actors is None else len(pc_actors)
        )
    else:
        mesh_actors, mesh_actor_ids, mesh_tree = None, None, None

    totel_actors = pc_actors + mesh_actors
    totel_actor_ids = pc_actor_ids + mesh_actor_ids
    totel_tree = pc_tree + mesh_tree
    return totel_actors, totel_actor_ids, totel_tree


def drosophila_actors(
    pc_models: Optional[list] = None,
    pc_model_ids: Optional[list] = None,
    pc_added_kwargs: Optional[dict] = None,
    mesh_models: Optional[list] = None,
    mesh_model_ids: Optional[list] = None,
    mesh_added_kwargs: Optional[dict] = None,
    **kwargs
):

    # Generate actors
    pc_kwargs = dict(model_style="points", model_size=5)
    if not (pc_added_kwargs is None):
        pc_kwargs.update(pc_added_kwargs)
    mesh_kwargs = dict(opacity=0.5, model_style="surface")
    if not (mesh_added_kwargs is None):
        mesh_kwargs.update(mesh_added_kwargs)

    plotter, pc_actors, mesh_actors = generate_actors(
        pc_models=pc_models,
        mesh_models=mesh_models,
        pc_added_kwargs=pc_kwargs,
        mesh_added_kwargs=mesh_kwargs,
        **kwargs
    )

    # Generate the relationship tree of actors
    actors, actor_ids, tree = generate_actors_tree(
        pc_actors=pc_actors,
        pc_actor_ids=pc_model_ids,
        mesh_actors=mesh_actors,
        mesh_actor_ids=mesh_model_ids,
    )

    return plotter, actors, actor_ids, tree
