import os

import anndata as ad
import pyvista as pv
from typing import Optional
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def abstract_anndata(path: str, X_layer: str = "X"):
    adata = ad.read_h5ad(filename=path)
    if X_layer != "X":
        assert X_layer in adata.layers.keys(), "`X_layer` does not exist in `adata.layers`."
        adata.X = adata.layers[X_layer]
    return adata


def abstract_models(path: str, model_ids: Optional[list] = None):
    model_files = os.listdir(path=path)
    assert len(model_files) != 0, "There is no file under this path."

    models = [pv.read(filename=os.path.join(path, f)) for f in model_files]
    if model_ids is None:  # Cannot contain `-` and ` `.
        model_ids = [f"Model{i}" for i in range(len(models))]
    assert len(model_ids) == len(models), "The number of model_ids does not equal to that of models."

    return models, model_ids


def sample_dataset(
    path: str,
    X_layer: str = "X",
    pc_model_ids: Optional[list] = None,
    mesh_model_ids: Optional[list] = None,
):
    # Generate anndata object
    anndata_path = os.path.join(path, "h5ad")
    adata = abstract_anndata(path=os.path.join(anndata_path, os.listdir(path=anndata_path)[0]), X_layer=X_layer)

    # Generate point cloud models
    pc_models_path = os.path.join(path, "pc_models")
    pc_model_files = os.listdir(path=pc_models_path)
    if len(pc_model_files) == 0:
        pc_models, pc_model_ids = None, None
    else:
        if pc_model_ids is None:
            pc_model_ids = [f"PC_{str(i).split('_')[1]}" for i in pc_model_files]
        pc_models, pc_model_ids = abstract_models(path=pc_models_path, model_ids=pc_model_ids)

    # Generate mesh models
    mesh_models_path = os.path.join(path, "mesh_models")
    mesh_model_files = os.listdir(path=mesh_models_path)
    if len(mesh_model_files) == 0:
        mesh_models, mesh_model_ids = None, None
    else:
        if mesh_model_ids is None:
            mesh_model_ids = [f"Mesh_{str(i).split('_')[1]}" for i in mesh_model_files]
        mesh_models, mesh_model_ids = abstract_models(path=mesh_models_path, model_ids=mesh_model_ids)

    return adata, pc_models, pc_model_ids, mesh_models, mesh_model_ids

