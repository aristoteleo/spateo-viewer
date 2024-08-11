import random
from typing import List, Optional, Tuple, Union

import numpy as np
import ot
import torch
from anndata import AnnData

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def downsampling(
    models: Union[List[AnnData], AnnData],
    n_sampling: Optional[int] = 2000,
    sampling_method: str = "trn",
    spatial_key: str = "spatial",
) -> Union[List[AnnData], AnnData]:
    from dynamo.tools.sampling import sample

    sampling_models = models if isinstance(models, list) else [models]
    for sampling_model in sampling_models:
        if sampling_model.shape[0] > n_sampling:
            sampling = sample(
                arr=np.asarray(sampling_model.obs_names),
                n=n_sampling,
                method=sampling_method,
                X=sampling_model.obsm[spatial_key],
            )
            sampling_model = sampling_model[sampling, :]
        sampling_models.append(sampling_model)
    return sampling_models


############################
# Alignment based on PASTE #
############################


def paste_pairwise_align(
    sampleA: AnnData,
    sampleB: AnnData,
    layer: str = "X",
    genes: Optional[Union[list, np.ndarray]] = None,
    spatial_key: str = "spatial",
    alpha: float = 0.1,
    dissimilarity: str = "kl",
    G_init=None,
    a_distribution=None,
    b_distribution=None,
    norm: bool = False,
    numItermax: int = 200,
    numItermaxEmd: int = 100000,
    dtype: str = "float64",
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[np.ndarray, Optional[int]]:
    from .alignment_utils import align_preprocess, calc_exp_dissimilarity

    # Preprocessing
    (
        nx,
        type_as,
        new_samples,
        exp_matrices,
        spatial_coords,
        normalize_scale,
        normalize_mean_list,
    ) = align_preprocess(
        samples=[sampleA, sampleB],
        genes=genes,
        spatial_key=spatial_key,
        layer=layer,
        normalize_c=False,
        normalize_g=False,
        select_high_exp_genes=False,
        dtype=dtype,
        device=device,
        verbose=verbose,
    )

    # Calculate spatial distances
    coordsA, coordsB = spatial_coords[0], spatial_coords[1]
    D_A = ot.dist(coordsA, coordsA, metric="euclidean")
    D_B = ot.dist(coordsB, coordsB, metric="euclidean")

    # Calculate expression dissimilarity
    X_A, X_B = exp_matrices[0], exp_matrices[1]
    M = calc_exp_dissimilarity(X_A=X_A, X_B=X_B, dissimilarity=dissimilarity)

    # init distributions
    a = (
        np.ones((sampleA.shape[0],)) / sampleA.shape[0]
        if a_distribution is None
        else np.asarray(a_distribution)
    )
    b = (
        np.ones((sampleB.shape[0],)) / sampleB.shape[0]
        if b_distribution is None
        else np.asarray(b_distribution)
    )
    a = nx.from_numpy(a, type_as=type_as)
    b = nx.from_numpy(b, type_as=type_as)

    if norm:
        D_A /= nx.min(D_A[D_A > 0])
        D_B /= nx.min(D_B[D_B > 0])

    # Run OT
    constC, hC1, hC2 = ot.gromov.init_matrix(D_A, D_B, a, b, "square_loss")

    if G_init is None:
        G0 = a[:, None] * b[None, :]
    else:
        G_init = nx.from_numpy(G_init, type_as=type_as)
        G0 = (1 / nx.sum(G_init)) * G_init

    if ot.__version__ == "0.8.3dev":
        pi, log = ot.optim.cg(
            a,
            b,
            (1 - alpha) * M,
            alpha,
            lambda G: ot.gromov.gwloss(constC, hC1, hC2, G),
            lambda G: ot.gromov.gwggrad(constC, hC1, hC2, G),
            G0,
            armijo=False,
            C1=D_A,
            C2=D_B,
            constC=constC,
            numItermax=numItermax,
            numItermaxEmd=numItermaxEmd,
            log=True,
        )
    else:
        pi, log = ot.gromov.cg(
            a,
            b,
            (1 - alpha) * M,
            alpha,
            lambda G: ot.gromov.gwloss(constC, hC1, hC2, G),
            lambda G: ot.gromov.gwggrad(constC, hC1, hC2, G),
            G0,
            armijo=False,
            C1=D_A,
            C2=D_B,
            constC=constC,
            numItermax=numItermax,
            numItermaxEmd=numItermaxEmd,
            log=True,
        )

    pi = nx.to_numpy(pi)
    obj = nx.to_numpy(log["loss"][-1])
    if device != "cpu":
        torch.cuda.empty_cache()

    return pi, obj


def generalized_procrustes_analysis(X, Y, pi):
    tX = pi.sum(axis=1).dot(X)
    tY = pi.sum(axis=0).dot(Y)
    X = X - tX
    Y = Y - tY
    H = Y.T.dot(pi.T.dot(X))
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T.dot(U.T)
    Y = R.dot(Y.T).T
    mapping_dict = {"tX": tX, "tY": tY, "R": R}
    return X, Y, mapping_dict


def paste_transform(
    adata: AnnData,
    adata_ref: AnnData,
    spatial_key: str = "spatial",
    key_added: str = "align_spatial",
    mapping_key: str = "models_align",
) -> AnnData:
    """
    Align the space coordinates of the new model with the transformation matrix obtained from PASTE.

    Args:
        adata: The anndata object that need to be aligned.
        adata_ref: The anndata object that have been aligned by PASTE.
        spatial_key: The key in `.obsm` that corresponds to the raw spatial coordinates.
        key_added: ``.obsm`` key under which to add the aligned spatial coordinates.
        mapping_key: The key in `.uns` that corresponds to the alignment info from PASTE.

    Returns:
        adata: The anndata object that have been to be aligned.
    """

    assert mapping_key in adata_ref.uns_keys(), "`mapping_key` value is wrong."
    tX = adata_ref.uns[mapping_key]["tX"]
    tY = adata_ref.uns[mapping_key]["tY"]
    R = adata_ref.uns[mapping_key]["R"]

    adata_coords = adata.obsm[spatial_key].copy() - tY
    adata.obsm[key_added] = R.dot(adata_coords.T).T + tX
    return adata


def paste_align(
    models: List[AnnData],
    spatial_key: str = "spatial",
    key_added: str = "align_spatial",
    n_sampling: Optional[int] = 2000,
    sampling_method: str = "random",
    alpha: float = 0.1,
    numItermax: int = 200,
    numItermaxEmd: int = 100000,
    dtype: str = "float64",
    device: str = "cpu",
    verbose: bool = False,
    **kwargs,
) -> List[AnnData]:
    """
    Align spatial coordinates of models.

    Args:
        models: List of models (AnnData Object).
        spatial_key: The key in ``.obsm`` that corresponds to the raw spatial coordinate.
        key_added: ``.obsm`` key under which to add the aligned spatial coordinates.
        n_sampling: When ``models_ref`` is None, new data containing n_sampling coordinate points will be automatically generated for alignment.
        sampling_method: The method to sample data points, can be one of ``["trn", "kmeans", "random"]``.
        alpha: Alignment tuning parameter. Note: 0 <= alpha <= 1.

               When ``alpha = 0`` only the gene expression data is taken into account,
               while when ``alpha =1`` only the spatial coordinates are taken into account.
        numItermax: Max number of iterations for cg during FGW-OT.
        numItermaxEmd: Max number of iterations for emd during FGW-OT.
        dtype: The floating-point number type. Only ``float32`` and ``float64``.
        device: Equipment used to run the program. You can also set the specified GPU for running. ``E.g.: '0'``.
        verbose: If ``True``, print progress updates.
        **kwargs: Additional parameters that will be passed to ``pairwise_align`` function.

    Returns:
        align_models: List of models (AnnData Object) after alignment.
    """
    models_sampling = [model.copy() for model in models]
    mean_cells = np.mean([model.shape[0] for model in models_sampling])
    to_sampling = 0 < n_sampling <= mean_cells and n_sampling is not None
    if to_sampling:
        models_ref = downsampling(
            models=models_sampling,
            n_sampling=n_sampling,
            sampling_method=sampling_method,
            spatial_key=spatial_key,
        )
    else:
        models_ref = models_sampling

    for m in models_ref:
        m.obsm[key_added] = m.obsm[spatial_key]

    pis = []
    align_models_ref = [model.copy() for model in models_ref]
    for i in range(len(align_models_ref) - 1):
        modelA = align_models_ref[i]
        modelB = align_models_ref[i + 1]

        # Calculate and returns optimal alignment of two models.
        pi, _ = paste_pairwise_align(
            sampleA=modelA.copy(),
            sampleB=modelB.copy(),
            layer="X",
            genes=None,
            spatial_key=key_added,
            alpha=alpha,
            numItermax=numItermax,
            numItermaxEmd=numItermaxEmd,
            dtype=dtype,
            device=device,
            verbose=verbose,
            **kwargs,
        )
        pis.append(pi)

        # Calculate new coordinates of two models
        modelA_coords, modelB_coords, mapping_dict = generalized_procrustes_analysis(
            X=modelA.obsm[key_added], Y=modelB.obsm[key_added], pi=pi
        )

        modelA.obsm[key_added] = modelA_coords
        modelB.obsm[key_added] = modelB_coords
        modelB.uns["models_align"] = mapping_dict

    align_models = []
    if to_sampling:
        for i, (align_model_ref, model) in enumerate(zip(align_models_ref, models)):
            align_model = model.copy()
            if i == 0:
                align_model.obsm[key_added] = align_model.obsm[spatial_key]
            else:
                align_model = paste_transform(
                    adata=align_model,
                    adata_ref=align_model_ref,
                    spatial_key=spatial_key,
                    key_added=key_added,
                    mapping_key="models_align",
                )
            align_models.append(align_model)
    else:
        for align_model_ref in align_models_ref:
            align_models.append(align_model_ref)

    return align_models_ref


#############################
# Alignment based on Morpho #
#############################


def con_K(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    beta: Union[int, float] = 0.01,
) -> Union[np.ndarray, torch.Tensor]:
    from .alignment_utils import cal_dist

    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."
    nx = ot.backend.get_backend(X, Y)

    K = cal_dist(X, Y)
    K = nx.exp(-beta * K)
    return K


def get_P(
    XnAHat: Union[np.ndarray, torch.Tensor],
    XnB: Union[np.ndarray, torch.Tensor],
    sigma2: Union[int, float, np.ndarray, torch.Tensor],
    beta2: Union[int, float, np.ndarray, torch.Tensor],
    alpha: Union[np.ndarray, torch.Tensor],
    gamma: Union[float, np.ndarray, torch.Tensor],
    Sigma: Union[np.ndarray, torch.Tensor],
    GeneDistMat: Union[np.ndarray, torch.Tensor],
    SpatialDistMat: Union[np.ndarray, torch.Tensor],
    samples_s: Optional[List[float]] = None,
    outlier_variance: float = None,
):
    from .alignment_utils import _data, _mul, _pi, _power, _prod, _unsqueeze

    assert (
        XnAHat.shape[1] == XnB.shape[1]
    ), "XnAHat and XnB do not have the same number of features."
    assert (
        XnAHat.shape[0] == alpha.shape[0]
    ), "XnAHat and alpha do not have the same length."
    assert (
        XnAHat.shape[0] == Sigma.shape[0]
    ), "XnAHat and Sigma do not have the same length."

    nx = ot.backend.get_backend(XnAHat, XnB)
    NA, NB, D = XnAHat.shape[0], XnB.shape[0], XnAHat.shape[1]
    if samples_s is None:
        samples_s = nx.maximum(
            _prod(nx)(nx.max(XnAHat, axis=0) - nx.min(XnAHat, axis=0)),
            _prod(nx)(nx.max(XnB, axis=0) - nx.min(XnB, axis=0)),
        )
    outlier_s = samples_s * NA
    if outlier_variance is None:
        exp_SpatialMat = nx.exp(-SpatialDistMat / (2 * sigma2))
    else:
        exp_SpatialMat = nx.exp(-SpatialDistMat / (2 * sigma2 / outlier_variance))
    spatial_term1 = nx.einsum(
        "ij,i->ij",
        exp_SpatialMat,
        (_mul(nx)(alpha, nx.exp(-Sigma / sigma2))),
    )
    spatial_outlier = (
        _power(nx)((2 * _pi(nx) * sigma2), _data(nx, D / 2, XnAHat))
        * (1 - gamma)
        / (gamma * outlier_s)
    )
    spatial_term2 = spatial_outlier + nx.einsum("ij->j", spatial_term1)
    spatial_P = spatial_term1 / _unsqueeze(nx)(spatial_term2, 0)
    spatial_inlier = 1 - spatial_outlier / (
        spatial_outlier + nx.einsum("ij->j", exp_SpatialMat)
    )
    term1 = nx.einsum(
        "ij,i->ij",
        _mul(nx)(
            nx.exp(-SpatialDistMat / (2 * sigma2)), nx.exp(-GeneDistMat / (2 * beta2))
        ),
        (_mul(nx)(alpha, nx.exp(-Sigma / sigma2))),
    )
    P = term1 / (_unsqueeze(nx)(nx.einsum("ij->j", term1), 0) + 1e-8)
    P = nx.einsum("j,ij->ij", spatial_inlier, P)

    term1 = nx.einsum(
        "ij,i->ij",
        nx.exp(-SpatialDistMat / (2 * sigma2)),
        (_mul(nx)(alpha, nx.exp(-Sigma / sigma2))),
    )
    sigma2_P = term1 / (_unsqueeze(nx)(nx.einsum("ij->j", term1), 0) + 1e-8)
    sigma2_P = nx.einsum("j,ij->ij", spatial_inlier, sigma2_P)
    return P, spatial_P, sigma2_P


def BA_align(
    sampleA: AnnData,
    sampleB: AnnData,
    spatial_key: str = "spatial",
    dissimilarity: str = "kl",
    max_iter: int = 200,
    max_outlier_variance: int = 20,
    lambdaVF: Union[int, float] = 1e2,
    beta: Union[int, float] = 0.01,
    K: Union[int, float] = 15,
    normalize_c: bool = True,
    normalize_g: bool = True,
    select_high_exp_genes: Union[bool, float, int] = False,
    dtype: str = "float64",
    device: str = "cpu",
    inplace: bool = False,
    nn_init: bool = True,
    batch_size: int = 1000,
):
    from .alignment_utils import (
        _data,
        _dot,
        _identity,
        _linalg,
        _pinv,
        _power,
        _psi,
        _randperm,
        _roll,
        _unique,
        align_preprocess,
        cal_dist,
        calc_exp_dissimilarity,
        coarse_rigid_alignment,
        empty_cache,
        get_optimal_R,
    )

    # Preprocessing
    empty_cache(device=device)
    normalize_g = False if dissimilarity == "kl" else normalize_g
    sampleA, sampleB = (
        (sampleA, sampleB) if inplace else (sampleA.copy(), sampleB.copy())
    )
    (
        nx,
        type_as,
        new_samples,
        exp_matrices,
        spatial_coords,
        normalize_scale,
        normalize_mean_list,
    ) = align_preprocess(
        samples=[sampleA, sampleB],
        layer="X",
        genes=None,
        spatial_key=spatial_key,
        normalize_c=normalize_c,
        normalize_g=normalize_g,
        select_high_exp_genes=select_high_exp_genes,
        dtype=dtype,
        device=device,
        verbose=False,
    )

    coordsA, coordsB = spatial_coords[1], spatial_coords[0]
    X_A, X_B = exp_matrices[1], exp_matrices[0]
    GeneDistMat = calc_exp_dissimilarity(X_A=X_A, X_B=X_B, dissimilarity=dissimilarity)
    del spatial_coords, exp_matrices

    NA, NB, D, G = coordsA.shape[0], coordsB.shape[0], coordsA.shape[1], X_A.shape[1]

    if nn_init:
        coordsA, inlier_A, inlier_B, inlier_P, init_R, init_t = coarse_rigid_alignment(
            coordsA, coordsB, X_A, X_B, dissimilarity=dissimilarity, top_K=10
        )
        empty_cache(device=device)
        coordsA = _data(nx, coordsA, type_as)
        inlier_A = _data(nx, inlier_A, type_as)
        inlier_B = _data(nx, inlier_B, type_as)
        inlier_P = _data(nx, inlier_P, type_as)
        init_R = _data(nx, init_R, type_as)
        init_t = _data(nx, init_t, type_as)
    else:
        init_R = _identity(nx, D, type_as)
        init_t = _data(nx, nx.zeros((D)), type_as)

    # Random select control points
    Unique_coordsA = _unique(nx, coordsA, 0)
    idx = random.sample(range(Unique_coordsA.shape[0]), min(K, Unique_coordsA.shape[0]))
    ctrl_pts = Unique_coordsA[idx, :]
    K = ctrl_pts.shape[0]

    # construct the kernel
    GammaSparse = con_K(ctrl_pts, ctrl_pts, beta)
    U = con_K(coordsA, ctrl_pts, beta)

    # initialize parameters
    kappa = nx.ones((NA), type_as=type_as)
    alpha = nx.ones((NA), type_as=type_as)
    VnA = nx.zeros(coordsA.shape, type_as=type_as)

    gamma, gamma_a, gamma_b = (
        _data(nx, 0.5, type_as),
        _data(nx, 1.0, type_as),
        _data(nx, 1.0, type_as),
    )
    SigmaDiag = nx.zeros((NA), type_as=type_as)
    XAHat, RnA = coordsA, coordsA
    SpatialDistMat = cal_dist(XAHat, coordsB)

    sigma2 = 0.1 * nx.sum(SpatialDistMat) / (D * NA * NB)  # 2 for 3D

    s = _data(nx, 1, type_as)
    R = _identity(nx, D, type_as)
    minGeneDistMat = nx.min(GeneDistMat, 1)
    # Automatically determine the value of beta2
    beta2_end = nx.max(minGeneDistMat) / 5
    beta2 = (
        minGeneDistMat[nx.argsort(minGeneDistMat)[int(GeneDistMat.shape[0] * 0.05)]] / 5
    )
    del minGeneDistMat
    # The value of beta2 becomes progressively larger
    beta2 = nx.maximum(beta2, _data(nx, 1e-2, type_as))
    beta2_decrease = _power(nx)(beta2_end / beta2, 1 / (50))
    # If partial alignment, use smaller spatial variance to reduce tails
    outlier_variance = 1
    outlier_variance_decrease = _power(nx)(
        _data(nx, max_outlier_variance, type_as), 1 / (max_iter / 2)
    )

    SVI_deacy = _data(nx, 10.0, type_as)
    batch_size = min(max(int(NB / 10), batch_size), NB)
    randomidx = _randperm(nx)(NB)
    randIdx = randomidx[:batch_size]
    randcoordsB = coordsB[randIdx, :]  # batch_size x D
    randGeneDistMat = GeneDistMat[:, randIdx]  # NA x batch_size
    SpatialDistMat = SpatialDistMat[:, randIdx]  # NA x batch_size
    Sp, Sp_spatial, Sp_sigma2 = 0, 0, 0
    SigmaInv = nx.zeros((K, K), type_as=type_as)  # K x K
    PXB_term = nx.zeros((NA, D), type_as=type_as)  # NA x D

    iteration = range(max_iter)
    for iter in iteration:
        step_size = nx.minimum(_data(nx, 1.0, type_as), SVI_deacy / (iter + 1.0))
        P, spatial_P, sigma2_P = get_P(
            XnAHat=XAHat,
            XnB=randcoordsB,
            sigma2=sigma2,
            beta2=beta2,
            alpha=alpha,
            gamma=gamma,
            Sigma=SigmaDiag,
            GeneDistMat=randGeneDistMat,
            SpatialDistMat=SpatialDistMat,
            outlier_variance=outlier_variance,
        )

        if iter > 5:
            if beta2_decrease < 1:
                beta2 = nx.maximum(beta2 * beta2_decrease, beta2_end)
            else:
                beta2 = nx.minimum(beta2 * beta2_decrease, beta2_end)
            outlier_variance = nx.minimum(
                outlier_variance * outlier_variance_decrease, max_outlier_variance
            )
        K_NA = nx.einsum("ij->i", P)
        K_NB = nx.einsum("ij->j", P)
        K_NA_spatial = nx.einsum("ij->i", spatial_P)
        K_NA_sigma2 = nx.einsum("ij->i", sigma2_P)

        # Update gamma
        Sp = step_size * nx.einsum("ij->", P) + (1 - step_size) * Sp
        Sp_spatial = (
            step_size * nx.einsum("ij->", spatial_P) + (1 - step_size) * Sp_spatial
        )
        Sp_sigma2 = (
            step_size * nx.einsum("ij->", sigma2_P) + (1 - step_size) * Sp_sigma2
        )
        gamma = nx.exp(
            _psi(nx)(gamma_a + Sp_spatial) - _psi(nx)(gamma_a + gamma_b + batch_size)
        )

        gamma = _data(nx, 0.99, type_as) if gamma > 0.99 else gamma
        gamma = _data(nx, 0.01, type_as) if gamma < 0.01 else gamma

        # Update alpha
        alpha = nx.exp(
            _psi(nx)(kappa + K_NA_spatial) - _psi(nx)(kappa * NA + Sp_spatial)
        )

        # Update VnA
        if (sigma2 < 0.015 and s > 0.95) or (iter > 80):
            SigmaInv = (
                step_size
                * (
                    sigma2 * lambdaVF * GammaSparse
                    + _dot(nx)(U.T, nx.einsum("ij,i->ij", U, K_NA))
                )
                + (1 - step_size) * SigmaInv
            )
            term1 = _dot(nx)(_pinv(nx)(SigmaInv), U.T)
            PXB_term = (
                step_size
                * (_dot(nx)(P, randcoordsB) - nx.einsum("ij,i->ij", RnA, K_NA))
                + (1 - step_size) * PXB_term
            )
            Coff = _dot(nx)(term1, PXB_term)
            VnA = _dot(nx)(
                U,
                Coff,
            )
            SigmaDiag = sigma2 * nx.einsum("ij->i", nx.einsum("ij,ji->ij", U, term1))

        # Update R()
        lambdaReg = 1e0 * Sp / nx.sum(inlier_P)
        PXA, PVA, PXB = (
            _dot(nx)(K_NA, coordsA)[None, :],
            _dot(nx)(K_NA, VnA)[None, :],
            _dot(nx)(K_NB, randcoordsB)[None, :],
        )

        PCYC, PCXC = _dot(nx)(inlier_P.T, inlier_B), _dot(nx)(inlier_P.T, inlier_A)
        if iter > 1:
            t = (
                step_size
                * (
                    (
                        (PXB - PVA - _dot(nx)(PXA, R.T))
                        + 2 * lambdaReg * sigma2 * (PCYC - _dot(nx)(PCXC, R.T))
                    )
                    / (Sp + 2 * lambdaReg * sigma2 * nx.sum(inlier_P))
                )
                + (1 - step_size) * t
            )
        else:
            t = (
                (PXB - PVA - _dot(nx)(PXA, R.T))
                + 2 * lambdaReg * sigma2 * (PCYC - _dot(nx)(PCXC, R.T))
            ) / (Sp + 2 * lambdaReg * sigma2 * nx.sum(inlier_P))

        A = -(
            _dot(nx)(PXA.T, t)
            + _dot(nx)(
                coordsA.T,
                nx.einsum("ij,i->ij", VnA, K_NA) - _dot(nx)(P, randcoordsB),
            )
            + 2
            * lambdaReg
            * sigma2
            * (
                _dot(nx)(PCXC.T, t)
                - _dot(nx)(nx.einsum("ij,i->ij", inlier_A, inlier_P[:, 0]).T, inlier_B)
            )
        ).T

        svdU, svdS, svdV = _linalg(nx).svd(A)
        C = _identity(nx, D, type_as)
        C[-1, -1] = _linalg(nx).det(_dot(nx)(svdU, svdV))
        if iter > 1:
            R = step_size * (_dot(nx)(_dot(nx)(svdU, C), svdV)) + (1 - step_size) * R
        else:
            R = _dot(nx)(_dot(nx)(svdU, C), svdV)
        RnA = s * _dot(nx)(coordsA, R.T) + t
        XAHat = RnA + VnA

        # Update sigma2 and beta2
        SpatialDistMat = cal_dist(XAHat, randcoordsB)
        sigma2 = nx.maximum(
            (
                nx.einsum("ij,ij", sigma2_P, SpatialDistMat) / (D * Sp_sigma2)
                + nx.einsum("i,i", K_NA_sigma2, SigmaDiag) / Sp_sigma2
            ),
            _data(nx, 1e-3, type_as),
        )

        # Next batch
        if iter < max_iter - 1:
            randIdx = randomidx[:batch_size]
            randomidx = _roll(nx)(randomidx, batch_size)
            randcoordsB = coordsB[randIdx, :]
            randGeneDistMat = GeneDistMat[:, randIdx]  # NA x batch_size
            SpatialDistMat = cal_dist(XAHat, randcoordsB)
        empty_cache(device=device)

    # full data
    SpatialDistMat = cal_dist(XAHat, coordsB)
    P, _, _ = get_P(
        XnAHat=XAHat,
        XnB=coordsB,
        sigma2=sigma2,
        beta2=beta2,
        alpha=alpha,
        gamma=gamma,
        Sigma=SigmaDiag,
        GeneDistMat=GeneDistMat,
        SpatialDistMat=SpatialDistMat,
        outlier_variance=outlier_variance,
    )
    optimal_RnA, optimal_R, optimal_t = get_optimal_R(
        coordsA=coordsA,
        coordsB=coordsB,
        P=P,
        R_init=R,
    )

    empty_cache(device=device)
    output_R = _dot(nx)(optimal_R, init_R)
    output_t = (
        normalize_scale * (_dot(nx)(init_t, optimal_R.T) + optimal_t)
        + normalize_mean_list[0]
        - _dot(nx)(normalize_mean_list[1], output_R.T)
    )

    if str(device) != "cpu":
        output_R = np.asarray(output_R.cpu())
        output_t = np.asarray(output_t.cpu())
    return output_R, output_t


def morpho_align(
    models: List[AnnData],
    n_sampling: Optional[int] = 2000,
    sampling_method: str = "random",
    spatial_key: str = "spatial",
    key_added: str = "align_spatial",
    max_outlier_variance: int = 20,
    max_iter: int = 200,
    device: str = "cpu",
) -> List[AnnData]:
    """
    Continuous alignment of spatial transcriptomic coordinates with the reference models based on Morpho.

    Args:
        models: List of models (AnnData Object).
        n_sampling: When ``models_ref`` is None, new data containing n_sampling coordinate points will be automatically generated for alignment.
        sampling_method: The method to sample data points, can be one of ``["trn", "kmeans", "random"]``.
        spatial_key: The key in ``.obsm`` that corresponds to the raw spatial coordinate.
        key_added: ``.obsm`` key under which to add the aligned spatial coordinate.
        max_outlier_variance: Reduce the spatial variance to decrease Gaussian tails to achieve robustness to partial
                              alignment. Lower means less robust, but more accurate. Recommended setting from 1 to 50.
        max_iter: Max number of iterations for morpho alignment.
        device: Equipment used to run the program. You can also set the specified GPU for running. ``E.g.: '0'``.
        **kwargs: Additional parameters that will be passed to ``BA_align`` function.

    Returns:
        align_models: List of models (AnnData Object) after alignment.
    """
    from concurrent.futures import ProcessPoolExecutor

    models_sampling = [model.copy() for model in models]
    mean_cells = np.mean([model.shape[0] for model in models_sampling])
    to_sampling = 0 < n_sampling <= mean_cells and n_sampling is not None

    if to_sampling:
        models_ref = downsampling(
            models=models_sampling,
            n_sampling=n_sampling,
            sampling_method=sampling_method,
            spatial_key=spatial_key,
        )
    else:
        models_ref = models_sampling

    models_ref_A = models_ref[:-1]
    models_ref_B = models_ref[1:]
    align_models = [model.copy() for model in models]
    align_models[0].obsm[key_added] = align_models[0].obsm[spatial_key]
    align_Rotation, align_translation = [], []
    for modelA, modelB in zip(models_ref_A, models_ref_B):
        output_R, output_t = BA_align(
            modelA,
            modelB,
            spatial_key=spatial_key,
            max_iter=max_iter,
            max_outlier_variance=max_outlier_variance,
            device=device,
        )
        align_Rotation.append(output_R)
        align_translation.append(output_t)

    cur_R = np.eye(2)
    cur_t = np.zeros(2)
    for i in range(len(models) - 1):
        cur_t = align_translation[i] @ cur_R.T + cur_t
        cur_R = align_Rotation[i] @ cur_R
        align_models[i + 1].obsm[key_added] = (
            align_models[i + 1].obsm[spatial_key] @ cur_R.T + cur_t
        )
    return align_models
