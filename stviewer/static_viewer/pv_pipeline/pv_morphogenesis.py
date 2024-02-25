from typing import Literal, Optional

import numpy as np
from anndata import AnnData
from pyvista import PolyData
from scipy.integrate import odeint


def morphogenesis(
    source_adata: AnnData,
    source_pc_model: PolyData,
    target_adata: Optional[AnnData] = None,
    mapping_method: Literal["GP", "OT"] = "GP",
    mapping_factor: float = 0.2,
    mapping_device: str = "cpu",
    morphofield_factor: int = 3000,
    morphopath_t_end: int = 10000,
    morphopath_sampling: int = 500,
):
    try:
        import spateo as st
    except ImportError:
        raise ImportError(
            "You need to install the package `spateo`. "
            "\nInstall spateo via `pip install spateo-release`."
        )
    try:
        from dynamo.vectorfield import SvcVectorField
    except ImportError:
        raise ImportError(
            "You need to install the package `dynamo`. "
            "\nInstall dynamo via `pip install dynamo-release`."
        )

    # Preprocess
    _obs_index = source_pc_model.point_data["obs_index"]
    source_adata = source_adata[_obs_index, :]

    # 3D mapping and morphofield
    if mapping_method == "OT":
        if not (target_adata is None):
            _ = st.tdr.cell_directions(
                adataA=source_adata,
                adataB=target_adata,
                numItermaxEmd=2000000,
                spatial_key="spatial",
                key_added="cells_mapping",
                alpha=mapping_factor,
                device=mapping_device,
                inplace=True,
            )

        if "V_cells_mapping" not in source_adata.obsm.keys():
            raise ValueError("You need to add the target anndata object. ")

        st.tdr.morphofield_sparsevfc(
            adata=source_adata,
            spatial_key="spatial",
            V_key="V_cells_mapping",
            key_added="VecFld_morpho",
            NX=None,
            inplace=True,
        )
    elif mapping_method == "GP":
        if not (target_adata is None):
            align_models, _, _ = st.align.morpho_align_sparse(
                models=[target_adata.copy(), source_adata.copy()],
                spatial_key="spatial",
                key_added="mapping_spatial",
                device=mapping_device,
                mode="SN-S",
                max_iter=200,
                partial_robust_level=1,
                beta=0.1,  # nonrigid,
                beta2_end=mapping_factor,  # low beta2_end, high expression similarity
                lambdaVF=1,
                K=200,
                SVI_mode=True,
                use_sparse=True,
            )
            source_adata = align_models[1].copy()

        if "VecFld_morpho" not in source_adata.uns.keys():
            raise ValueError("You need to add the target anndata object. ")

        st.tdr.morphofield_gp(
            adata=source_adata,
            spatial_key="spatial",
            vf_key="VecFld_morpho",
            NX=np.asarray(source_adata.obsm["spatial"]),
            inplace=True,
        )

    # construct morphofield model
    source_adata.obs["V_z"] = source_adata.uns["VecFld_morpho"]["V"][:, 2].flatten()
    source_pc_model.point_data["vectors"] = source_adata.uns["VecFld_morpho"]["V"]
    source_pc_model.point_data["V_Z"] = source_pc_model.point_data["vectors"][
        :, 2
    ].flatten()

    pc_vectors, _ = st.tdr.construct_field(
        model=source_pc_model,
        vf_key="vectors",
        arrows_scale_key="vectors",
        n_sampling=None,
        factor=morphofield_factor,
        key_added="obs_index",
        label=source_pc_model.point_data["obs_index"],
    )

    # construct morphopath model
    st.tdr.morphopath(
        adata=source_adata,
        vf_key="VecFld_morpho",
        key_added="fate_morpho",
        t_end=morphopath_t_end,
        interpolation_num=20,
        cores=10,
    )
    trajectory_model, _ = st.tdr.construct_trajectory(
        adata=source_adata,
        fate_key="fate_morpho",
        n_sampling=morphopath_sampling,
        sampling_method="random",
        key_added="obs_index",
        label=np.asarray(source_adata.obs.index),
    )

    # morphometric features
    st.tdr.morphofield_acceleration(
        adata=source_adata, vf_key="VecFld_morpho", key_added="acceleration"
    )
    st.tdr.morphofield_curvature(
        adata=source_adata, vf_key="VecFld_morpho", key_added="curvature"
    )
    st.tdr.morphofield_curl(
        adata=source_adata, vf_key="VecFld_morpho", key_added="curl"
    )
    st.tdr.morphofield_torsion(
        adata=source_adata, vf_key="VecFld_morpho", key_added="torsion"
    )
    st.tdr.morphofield_divergence(
        adata=source_adata, vf_key="VecFld_morpho", key_added="divergence"
    )

    source_pc_index = source_pc_model.point_data["obs_index"]
    source_pc_model.point_data["acceleration"] = np.asarray(
        source_adata[np.asarray(source_pc_index)].obs["acceleration"]
    )
    source_pc_model.point_data["curvature"] = np.asarray(
        source_adata[np.asarray(source_pc_index)].obs["curvature"]
    )
    source_pc_model.point_data["curl"] = np.asarray(
        source_adata[np.asarray(source_pc_index)].obs["curl"]
    )
    source_pc_model.point_data["torsion"] = np.asarray(
        source_adata[np.asarray(source_pc_index)].obs["torsion"]
    )
    source_pc_model.point_data["divergence"] = np.asarray(
        source_adata[np.asarray(source_pc_index)].obs["divergence"]
    )

    pc_vectors_index = pc_vectors.point_data["obs_index"]
    pc_vectors.point_data["V_Z"] = np.asarray(
        source_adata[np.asarray(pc_vectors_index)].obs["V_z"]
    )
    pc_vectors.point_data["acceleration"] = np.asarray(
        source_adata[np.asarray(pc_vectors_index)].obs["acceleration"]
    )
    pc_vectors.point_data["curvature"] = np.asarray(
        source_adata[np.asarray(pc_vectors_index)].obs["curvature"]
    )
    pc_vectors.point_data["curl"] = np.asarray(
        source_adata[np.asarray(pc_vectors_index)].obs["curl"]
    )
    pc_vectors.point_data["torsion"] = np.asarray(
        source_adata[np.asarray(pc_vectors_index)].obs["torsion"]
    )
    pc_vectors.point_data["divergence"] = np.asarray(
        source_adata[np.asarray(pc_vectors_index)].obs["divergence"]
    )

    trajectory_index = trajectory_model.point_data["obs_index"]
    trajectory_model.point_data["V_Z"] = np.asarray(
        source_adata[np.asarray(trajectory_index)].obs["V_z"]
    )
    trajectory_model.point_data["acceleration"] = np.asarray(
        source_adata[np.asarray(trajectory_index)].obs["acceleration"]
    )
    trajectory_model.point_data["curvature"] = np.asarray(
        source_adata[np.asarray(trajectory_index)].obs["curvature"]
    )
    trajectory_model.point_data["curl"] = np.asarray(
        source_adata[np.asarray(trajectory_index)].obs["curl"]
    )
    trajectory_model.point_data["torsion"] = np.asarray(
        source_adata[np.asarray(trajectory_index)].obs["torsion"]
    )
    trajectory_model.point_data["divergence"] = np.asarray(
        source_adata[np.asarray(trajectory_index)].obs["divergence"]
    )

    # cell stages of animation
    t_ind = np.asarray(list(source_adata.uns["fate_morpho"]["t"].keys()), dtype=int)
    t_sort_ind = np.argsort(t_ind)
    t = np.asarray(list(source_adata.uns["fate_morpho"]["t"].values()))[t_sort_ind]
    flats = np.unique([int(item) for sublist in t for item in sublist])
    flats = np.hstack((0, flats))
    flats = np.sort(flats) if 3000 is None else np.sort(flats[flats <= 3000])
    time_vec = np.logspace(0, np.log10(max(flats) + 1), 100) - 1
    vf = SvcVectorField()
    vf.from_adata(source_adata, basis="morpho")
    f = lambda x, _: vf.func(x)
    displace = lambda x, dt: odeint(f, x, [0, dt])

    init_states = source_adata.uns["fate_morpho"]["init_states"]
    pts = [i.tolist() for i in init_states]
    stages_X = [source_adata.obs.index.tolist()]
    for i in range(100):
        pts = [displace(cur_pts, time_vec[i])[1].tolist() for cur_pts in pts]
        stages_X.append(pts)

    return source_pc_model, pc_vectors, trajectory_model, stages_X
