from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import pyvista as pv
from anndata import AnnData
from pyvista import PolyData


def RNAvelocity(
    adata: AnnData,
    pc_model: PolyData,
    layer: str = "X",
    basis_pca: str = "pca",
    basis_umap: str = "umap",
    data_preprocess: Literal[
        "False", "recipe_monocle", "pearson_residuals"
    ] = "recipe_monocle",
    harmony_debatch: bool = False,
    group_key: Optional[str] = None,
    n_neighbors: int = 30,
    n_pca_components: int = 30,
    n_vectors_downsampling: Optional[int] = None,
    vectors_size: Union[float, int] = 1,
):
    try:
        import dynamo as dyn
        from dynamo.preprocessing import Preprocessor
        from dynamo.tools.Markov import velocity_on_grid
    except ImportError:
        raise ImportError(
            "You need to install the package `dynamo`. "
            "\nInstall dynamo via `pip install dynamo-release`."
        )
    try:
        import harmonypy
    except ImportError:
        raise ImportError(
            "You need to install the package `harmonypy`. "
            "\nInstall harmonypy via `pip install harmonypy`."
        )

    # Preprocess
    _obs_index = pc_model.point_data["obs_index"]
    dyn_adata = adata[_obs_index, :].copy()
    dyn_adata.X = dyn_adata.X if layer == "X" else dyn_adata.layers[layer]

    # Data preprocess
    if basis_pca in dyn_adata.obsm.keys():
        if data_preprocess != "False":
            preprocessor = Preprocessor()
            preprocessor.preprocess_adata(adata, recipe="monocle")
        dyn.tl.reduceDimension(
            dyn_adata,
            basis=basis_pca,
            n_neighbors=n_neighbors,
            n_pca_components=n_pca_components,
        )
    else:
        if data_preprocess != "False":
            if data_preprocess == "recipe_monocle":
                preprocessor = Preprocessor()
                preprocessor.preprocess_adata(adata, recipe="monocle")
            elif data_preprocess == "pearson_residuals":
                preprocessor = Preprocessor()
                preprocessor.preprocess_adata(adata, recipe="pearson_residuals")
        dyn.tl.reduceDimension(
            dyn_adata,
            basis=basis_pca,
            n_neighbors=n_neighbors,
            n_pca_components=n_pca_components,
        )
        if harmony_debatch:
            harmony_out = harmonypy.run_harmony(
                dyn_adata.obsm[basis_pca], dyn_adata.obs, group_key, max_iter_harmony=20
            )
            dyn_adata.obsm[basis_pca] = harmony_out.Z_corr.T
            dyn.tl.reduceDimension(
                dyn_adata,
                X_data=dyn_adata.obsm[basis_pca],
                enforce=True,
                n_neighbors=n_neighbors,
                n_pca_components=n_pca_components,
            )

    # RNA velocity
    if basis_umap in dyn_adata.obsm.keys():
        if f"X_{basis_umap}" not in dyn_adata.obsm.keys():
            dyn_adata.obsm[f"X_{basis_umap}"] = dyn_adata.obsm[basis_umap]

    dyn.tl.dynamics(dyn_adata, model="stochastic", cores=3)
    dyn.tl.cell_velocities(
        dyn_adata,
        basis=basis_umap,
        method="pearson",
        other_kernels_dict={"transform": "sqrt"},
    )
    dyn.tl.cell_velocities(dyn_adata, basis=basis_pca)

    # Vectorfield
    dyn.vf.VectorField(dyn_adata, basis=basis_umap)
    dyn.vf.VectorField(dyn_adata, basis=basis_pca)

    # Pesudotime
    dyn.ext.ddhodge(dyn_adata, basis=basis_umap)
    dyn.ext.ddhodge(dyn_adata, basis=basis_pca)

    # Differnetial geometry
    dyn.vf.speed(dyn_adata, basis=basis_pca)
    dyn.vf.acceleration(dyn_adata, basis=basis_pca)
    dyn.vf.curvature(dyn_adata, basis=basis_pca)
    dyn.vf.curl(dyn_adata, basis=basis_pca)
    dyn.vf.divergence(dyn_adata, basis=basis_pca)

    # RNA velocity vectors model
    if n_vectors_downsampling in [None, "None", "none"]:
        ix_choice = np.arange(dyn_adata.shape[0])
    else:
        ix_choice = np.random.choice(
            np.arange(dyn_adata.shape[0]), size=n_vectors_downsampling, replace=False
        )

    X = dyn_adata.obsm[f"X_{basis_umap}"][:, [0, 1]]
    V = dyn_adata.obsm[f"velocity_{basis_umap}"][:, [0, 1]]
    X, V = X[ix_choice, :], V[ix_choice, :]
    if X.shape[1] == 2:
        df = pd.DataFrame(
            {
                "x": X[:, 0],
                "y": X[:, 1],
                "z": np.zeros(shape=(X.shape[0])),
                "u": V[:, 0],
                "v": V[:, 1],
                "w": np.zeros(shape=(V.shape[0])),
            }
        )
    else:
        df = pd.DataFrame(
            {
                "x": X[:, 0],
                "y": X[:, 1],
                "z": X[:, 2],
                "u": V[:, 0],
                "v": V[:, 1],
                "w": V[:, 2],
            }
        )
    df = df.iloc[ix_choice, :]

    x0, x1, x2 = df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]
    v0, v1, v2 = df.iloc[:, 3], df.iloc[:, 4], df.iloc[:, 5]

    point_cloud = pv.PolyData(np.column_stack((x0.values, x1.values, x2.values)))
    point_cloud["vectors"] = np.column_stack((v0.values, v1.values, v2.values))
    point_cloud.point_data["obs_index"] = dyn_adata.obs.index.tolist()
    vectors = point_cloud.glyph(orient="vectors", factor=vectors_size)
    vectors.point_data[f"{basis_umap}_ddhodge_potential"] = np.asarray(
        dyn_adata[np.asarray(vectors.point_data["obs_index"])].obs[
            f"{basis_umap}_ddhodge_potential"
        ]
    )
    vectors.point_data[f"{basis_pca}_ddhodge_potential"] = np.asarray(
        dyn_adata[np.asarray(vectors.point_data["obs_index"])].obs[
            f"{basis_pca}_ddhodge_potential"
        ]
    )
    vectors.point_data[f"speed_{basis_pca}"] = np.asarray(
        dyn_adata[np.asarray(vectors.point_data["obs_index"])].obs[f"speed_{basis_pca}"]
    )
    vectors.point_data[f"acceleration_{basis_pca}"] = np.asarray(
        dyn_adata[np.asarray(vectors.point_data["obs_index"])].obs[
            f"acceleration_{basis_pca}"
        ]
    )
    vectors.point_data[f"divergence_{basis_pca}"] = np.asarray(
        dyn_adata[np.asarray(vectors.point_data["obs_index"])].obs[
            f"divergence_{basis_pca}"
        ]
    )
    vectors.point_data[f"curvature_{basis_pca}"] = np.asarray(
        dyn_adata[np.asarray(vectors.point_data["obs_index"])].obs[
            f"curvature_{basis_pca}"
        ]
    )
    vectors.point_data[f"curl_{basis_pca}"] = np.asarray(
        dyn_adata[np.asarray(vectors.point_data["obs_index"])].obs[f"curl_{basis_pca}"]
    )

    pc_model.point_data[f"{basis_umap}_ddhodge_potential"] = np.asarray(
        dyn_adata[np.asarray(pc_model.point_data["obs_index"])].obs[
            f"{basis_umap}_ddhodge_potential"
        ]
    )
    pc_model.point_data[f"{basis_pca}_ddhodge_potential"] = np.asarray(
        dyn_adata[np.asarray(pc_model.point_data["obs_index"])].obs[
            f"{basis_pca}_ddhodge_potential"
        ]
    )
    pc_model.point_data[f"speed_{basis_pca}"] = np.asarray(
        dyn_adata[np.asarray(pc_model.point_data["obs_index"])].obs[
            f"speed_{basis_pca}"
        ]
    )
    pc_model.point_data[f"acceleration_{basis_pca}"] = np.asarray(
        dyn_adata[np.asarray(pc_model.point_data["obs_index"])].obs[
            f"acceleration_{basis_pca}"
        ]
    )
    pc_model.point_data[f"divergence_{basis_pca}"] = np.asarray(
        dyn_adata[np.asarray(pc_model.point_data["obs_index"])].obs[
            f"divergence_{basis_pca}"
        ]
    )
    pc_model.point_data[f"curvature_{basis_pca}"] = np.asarray(
        dyn_adata[np.asarray(pc_model.point_data["obs_index"])].obs[
            f"curvature_{basis_pca}"
        ]
    )
    pc_model.point_data[f"curl_{basis_pca}"] = np.asarray(
        dyn_adata[np.asarray(pc_model.point_data["obs_index"])].obs[f"curl_{basis_pca}"]
    )

    return pc_model, vectors
