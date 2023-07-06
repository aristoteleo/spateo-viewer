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
    mm_models: Optional[list] = None,
    mm_model_ids: Optional[list] = None,
    pc_added_kwargs: Optional[dict] = None,
    mesh_added_kwargs: Optional[dict] = None,
    mm_added_kwargs: Optional[dict] = None,
):
    # Generate actors for pc models
    pc_kwargs = dict(model_style="points", model_size=8)
    if not (pc_added_kwargs is None):
        pc_kwargs.update(pc_added_kwargs)
    if not (pc_models is None):
        pc_actors = [
            add_single_model(plotter=plotter, model=model, **pc_kwargs)
            for model in pc_models
        ]
    else:
        pc_actors = None

    # Generate actors for mesh models
    mesh_kwargs = dict(opacity=0.5, model_style="surface")
    if not (mesh_added_kwargs is None):
        mesh_kwargs.update(mesh_added_kwargs)
    if not (mesh_models is None):
        mesh_actors = [
            add_single_model(plotter=plotter, model=model, **mesh_kwargs)
            for model in mesh_models
        ]
    else:
        mesh_actors = None

    # Generate actors for morphometric models
    mm_kwargs = dict(opacity=0.3)
    if not (mm_added_kwargs is None):
        mm_kwargs.update(mm_added_kwargs)
    if not (mm_models is None):
        mm_actors = []
        mm_model_ids = (
            ["Trajectory"] * len(mm_models) if mm_model_ids is None else mm_model_ids
        )
        for model, model_id in zip(mm_models, mm_model_ids):
            if str(model_id).endswith("Trajectory"):
                _mm_actor = add_single_model(
                    plotter=plotter, model=model, model_style="wireframe", **mm_kwargs
                )
            else:
                _mm_actor = add_single_model(
                    plotter=plotter, model=model, model_style="surface", **mm_kwargs
                )
            _mm_actor.SetVisibility(False)
            mm_actors.append(_mm_actor)
    else:
        mm_actors = None
    return pc_actors, mesh_actors, mm_actors


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
            actors=mesh_actors,
            actor_names=mesh_actor_ids,
            base_id=0 if pc_actors is None else len(pc_actors),
        )
    else:
        mesh_actors, mesh_actor_ids, mesh_tree = None, None, None

    totel_actors = pc_actors + mesh_actors
    totel_actor_ids = pc_actor_ids + mesh_actor_ids
    totel_tree = pc_tree + mesh_tree
    return totel_actors, totel_actor_ids, totel_tree


def init_actors(plotter, path):
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
    pc_actors, mesh_actors, mm_actors = generate_actors(
        plotter=plotter,
        pc_models=pc_models,
        mesh_models=mesh_models,
        mm_models=mm_models,
        mm_model_ids=mm_model_ids,
    )

    # Generate the relationship tree of actors
    actors, actor_ids, tree = generate_actors_tree(
        pc_actors=pc_actors,
        pc_actor_ids=pc_model_ids,
        mesh_actors=mesh_actors,
        mesh_actor_ids=mesh_model_ids,
    )

    anndata_dir = os.path.join(path, "h5ad")
    anndata_path = os.path.join(anndata_dir, os.listdir(path=anndata_dir)[0])
    anndata_metrices = ["X"] + [i for i in adata.layers.keys()]
    return (
        anndata_path,
        anndata_metrices,
        actors,
        actor_ids,
        tree,
        mm_actors,
        mm_model_ids,
    )
