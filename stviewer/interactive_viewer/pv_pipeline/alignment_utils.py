import os
from typing import List, Optional, Tuple, Union

import numpy as np
import ot
import pandas as pd
import torch
from anndata import AnnData
from scipy.linalg import pinv
from scipy.sparse import issparse
from scipy.special import psi
from spateo.logging import logger_manager as lm

# Get the intersection of lists
intersect_lsts = lambda *lsts: list(set(lsts[0]).intersection(*lsts[1:]))

# Covert a sparse matrix into a dense np array
to_dense_matrix = lambda X: X.toarray() if issparse(X) else np.array(X)

# Returns the data matrix or representation
extract_data_matrix = lambda adata, rep: adata.X if rep is None else adata.layers[rep]


###########################
# Check data and computer #
###########################


def check_backend(device: str = "cpu", dtype: str = "float32", verbose: bool = True):
    """
    Check the proper backend for the device.

    Args:
        device: Equipment used to run the program. You can also set the specified GPU for running. E.g.: '0'.
        dtype: The floating-point number type. Only float32 and float64.
        verbose: If ``True``, print progress updates.

    Returns:
        backend: The proper backend.
        type_as: The type_as.device is the device used to run the program and the type_as.dtype is the floating-point number type.
    """
    if device == "cpu":
        backend = ot.backend.NumpyBackend()
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        if torch.cuda.is_available():
            torch.cuda.init()
            backend = ot.backend.TorchBackend()
        else:
            backend = ot.backend.NumpyBackend()
            if verbose:
                lm.main_info(
                    message="GPU is not available, resorting to torch cpu.",
                    indent_level=1,
                )
    if nx_torch(backend):
        type_as = (
            backend.__type_list__[-2]
            if dtype == "float32"
            else backend.__type_list__[-1]
        )
    else:
        type_as = (
            backend.__type_list__[0] if dtype == "float32" else backend.__type_list__[1]
        )
    return backend, type_as


def check_spatial_coords(sample: AnnData, spatial_key: str = "spatial") -> np.ndarray:
    """
    Check spatial coordinate information.

    Args:
        sample: An anndata object.
        spatial_key: The key in `.obsm` that corresponds to the raw spatial coordinates.

    Returns:
        The spatial coordinates.
    """
    coordinates = sample.obsm[spatial_key].copy()
    if isinstance(coordinates, pd.DataFrame):
        coordinates = coordinates.values

    return np.asarray(coordinates)


def check_exp(sample: AnnData, layer: str = "X") -> np.ndarray:
    """
    Check expression matrix.

    Args:
        sample: An anndata object.
        layer: The key in `.layers` that corresponds to the expression matrix.

    Returns:
        The expression matrix.
    """

    exp_martix = sample.X.copy() if layer == "X" else sample.layers[layer].copy()
    exp_martix = to_dense_matrix(exp_martix)
    return exp_martix


######################
# Data preprocessing #
######################


def filter_common_genes(*genes, verbose: bool = True) -> list:
    """
    Filters for the intersection of genes between all samples.

    Args:
        genes: List of genes.
        verbose: If ``True``, print progress updates.
    """

    common_genes = intersect_lsts(*genes)
    if len(common_genes) == 0:
        raise ValueError("The number of common gene between all samples is 0.")
    else:
        if verbose:
            lm.main_info(
                message=f"Filtered all samples for common genes. There are {(len(common_genes))} common genes.",
                indent_level=1,
            )
        return common_genes


def normalize_coords(
    coords: Union[List[np.ndarray or torch.Tensor], np.ndarray, torch.Tensor],
    nx: Union[
        ot.backend.TorchBackend, ot.backend.NumpyBackend
    ] = ot.backend.NumpyBackend,
    verbose: bool = True,
) -> Tuple[List[np.ndarray], float, List[np.ndarray]]:
    """Normalize the spatial coordinate.

    Args:
        coords: Spatial coordinate of sample.
        nx: The proper backend.
        verbose: If ``True``, print progress updates.
    """
    if type(coords) in [np.ndarray, torch.Tensor]:
        coords = [coords]

    normalize_scale = 0
    normalize_mean_list = []
    for i in range(len(coords)):
        normalize_mean = nx.einsum("ij->j", coords[i]) / coords[i].shape[0]
        normalize_mean_list.append(normalize_mean)
        coords[i] -= normalize_mean
        normalize_scale += nx.sqrt(
            nx.einsum("ij->", nx.einsum("ij,ij->ij", coords[i], coords[i]))
            / coords[i].shape[0]
        )

    normalize_scale /= len(coords)
    for i in range(len(coords)):
        coords[i] /= normalize_scale
    if verbose:
        lm.main_info(message=f"Coordinates normalization params:", indent_level=1)
        lm.main_info(message=f"Scale: {normalize_scale}.", indent_level=2)
        # lm.main_info(message=f"Mean:  {normalize_mean_list}", indent_level=2)
    return coords, normalize_scale, normalize_mean_list


def normalize_exps(
    matrices: List[np.ndarray or torch.Tensor],
    nx: Union[
        ot.backend.TorchBackend, ot.backend.NumpyBackend
    ] = ot.backend.NumpyBackend,
    verbose: bool = True,
) -> List[np.ndarray]:
    """Normalize the gene expression.

    Args:
        matrices: Gene expression of sample.
        nx: The proper backend.
        verbose: If ``True``, print progress updates.
    """
    if type(matrices) in [np.ndarray, torch.Tensor]:
        matrices = [matrices]
    normalize_scale = 0
    normalize_mean_list = []
    for i in range(len(matrices)):
        normalize_mean = nx.einsum("ij->j", matrices[i]) / matrices[i].shape[0]
        normalize_mean_list.append(normalize_mean)
        # coords[i] -= normalize_mean
        normalize_scale += nx.sqrt(
            nx.einsum("ij->", nx.einsum("ij,ij->ij", matrices[i], matrices[i]))
            / matrices[i].shape[0]
        )

    normalize_scale /= len(matrices)
    for i in range(len(matrices)):
        matrices[i] /= normalize_scale
    if verbose:
        lm.main_info(message=f"Gene expression normalization params:", indent_level=1)
        # lm.main_info(message=f"Mean: {normalize_mean}.", indent_level=2)
        lm.main_info(message=f"Scale: {normalize_scale}.", indent_level=2)

    return matrices


def align_preprocess(
    samples: List[AnnData],
    genes: Optional[Union[list, np.ndarray]] = None,
    spatial_key: str = "spatial",
    layer: str = "X",
    normalize_c: bool = False,
    normalize_g: bool = False,
    select_high_exp_genes: Union[bool, float, int] = False,
    dtype: str = "float64",
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[
    ot.backend.TorchBackend or ot.backend.NumpyBackend,
    torch.Tensor or np.ndarray,
    list,
    list,
    list,
    Optional[float],
    Optional[list],
]:
    """
    Data preprocessing before alignment.

    Args:
        samples: A list of anndata object.
        genes: Genes used for calculation. If None, use all common genes for calculation.
        spatial_key: The key in `.obsm` that corresponds to the raw spatial coordinates.
        layer: If `'X'`, uses ``sample.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``sample.layers[layer]``.
        normalize_c: Whether to normalize spatial coordinates.
        normalize_g: Whether to normalize gene expression.
        select_high_exp_genes: Whether to select genes with high differences in gene expression.
        dtype: The floating-point number type. Only float32 and float64.
        device: Equipment used to run the program. You can also set the specified GPU for running. E.g.: '0'.
        verbose: If ``True``, print progress updates.
    """

    # Determine if gpu or cpu is being used
    nx, type_as = check_backend(device=device, dtype=dtype)
    # Subset for common genes
    new_samples = [s.copy() for s in samples]
    all_samples_genes = [s.var.index for s in new_samples]
    common_genes = filter_common_genes(*all_samples_genes, verbose=verbose)
    common_genes = (
        common_genes if genes is None else intersect_lsts(common_genes, genes)
    )
    new_samples = [s[:, common_genes] for s in new_samples]

    # Gene expression matrix of all samples
    exp_matrices = [
        nx.from_numpy(check_exp(sample=s, layer=layer), type_as=type_as)
        for s in new_samples
    ]
    if not (select_high_exp_genes is False):
        # Select significance genes if select_high_exp_genes is True
        ExpressionData = _cat(nx=nx, x=exp_matrices, dim=0)

        ExpressionVar = _var(nx, ExpressionData, 0)
        exp_threshold = (
            10 if isinstance(select_high_exp_genes, bool) else select_high_exp_genes
        )
        EvidenceExpression = nx.where(ExpressionVar > exp_threshold)[0]
        exp_matrices = [
            exp_matrix[:, EvidenceExpression] for exp_matrix in exp_matrices
        ]
        if verbose:
            lm.main_info(
                message=f"Evidence expression number: {len(EvidenceExpression)}."
            )

    # Spatial coordinates of all samples
    spatial_coords = [
        nx.from_numpy(
            check_spatial_coords(sample=s, spatial_key=spatial_key), type_as=type_as
        )
        for s in new_samples
    ]
    coords_dims = nx.unique(_data(nx, [c.shape[1] for c in spatial_coords], type_as))
    # coords_dims = np.unique(np.asarray([c.shape[1] for c in spatial_coords]))
    assert (
        len(coords_dims) == 1
    ), "Spatial coordinate dimensions are different, please check again."

    normalize_scale, normalize_mean_list = None, None
    if normalize_c:
        spatial_coords, normalize_scale, normalize_mean_list = normalize_coords(
            coords=spatial_coords, nx=nx, verbose=verbose
        )
    if normalize_g:
        exp_matrices = normalize_exps(matrices=exp_matrices, nx=nx, verbose=verbose)

    return (
        nx,
        type_as,
        new_samples,
        exp_matrices,
        spatial_coords,
        normalize_scale,
        normalize_mean_list,
    )


def calc_exp_dissimilarity(
    X_A: Union[np.ndarray, torch.Tensor],
    X_B: Union[np.ndarray, torch.Tensor],
    dissimilarity: str = "kl",
    chunk_num: int = 1,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Calculate expression dissimilarity.
    Args:
        X_A: Gene expression matrix of sample A.
        X_B: Gene expression matrix of sample B.
        dissimilarity: Expression dissimilarity measure: ``'kl'`` or ``'euclidean'``.

    Returns:
        Union[np.ndarray, torch.Tensor]: The dissimilarity matrix of two feature samples.
    """
    nx = ot.backend.get_backend(X_A, X_B)

    assert dissimilarity in [
        "kl",
        "euclidean",
        "euc",
    ], "``dissimilarity`` value is wrong. Available ``dissimilarity`` are: ``'kl'``, ``'euclidean'`` and ``'euc'``."
    if dissimilarity.lower() == "kl":
        X_A = X_A + 0.01
        X_B = X_B + 0.01
        X_A = X_A / nx.sum(X_A, axis=1, keepdims=True)
        X_B = X_B / nx.sum(X_B, axis=1, keepdims=True)
    while True:
        try:
            if chunk_num == 1:
                DistMat = _dist(X_A, X_B, dissimilarity)
                break
            else:
                X_As = _chunk(nx, X_A, chunk_num, 0)
                X_Bs = _chunk(nx, X_B, chunk_num, 0)
                arr = []  # array for temporary storage of results
                for x_As in X_As:
                    arr2 = []
                    for x_Bs in X_Bs:
                        arr2.append(_dist(x_As, x_Bs, dissimilarity))
                    arr.append(nx.concatenate(arr2, axis=1))
                DistMat = nx.concatenate(arr, axis=0)
                break
        except:
            chunk_num = chunk_num * 2
            print("chunk more")
    return DistMat


def _dist(
    mat1: Union[np.ndarray, torch.Tensor],
    mat2: Union[np.ndarray, torch.Tensor],
    metric: str = "euc",
) -> Union[np.ndarray, torch.Tensor]:
    assert metric in [
        "euc",
        "euclidean",
        "kl",
    ], "``metric`` value is wrong. Available ``metric`` are: ``'euc'``, ``'euclidean'`` and ``'kl'``."
    nx = ot.backend.get_backend(mat1, mat2)
    if metric.lower() == "euc" or metric.lower() == "euclidean":
        distMat = (
            nx.sum(mat1**2, 1)[:, None]
            + nx.sum(mat2**2, 1)[None, :]
            - 2 * _dot(nx)(mat1, mat2.T)
        )
    else:
        distMat = (
            nx.sum(mat1 * nx.log(mat1), 1)[:, None]
            + nx.sum(mat2 * nx.log(mat2), 1)[None, :]
            - _dot(nx)(mat1, nx.log(mat2).T)
            - _dot(nx)(mat2, nx.log(mat1).T).T
        ) / 2
    return distMat


def cal_dist(
    X_A: Union[np.ndarray, torch.Tensor],
    X_B: Union[np.ndarray, torch.Tensor],
    use_gpu: bool = True,
    chunk_num: int = 1,
    return_gpu: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """Calculate the distance between two vectors

    Args:
        X_A (Union[np.ndarray, torch.Tensor]): The first input vector with shape n x d
        X_B (Union[np.ndarray, torch.Tensor]): The second input vector with shape m x d
        use_gpu (bool, optional): Whether to use GPU for chunk. Defaults to True.
        chunk_num (int, optional): The number of chunks. The larger the number, the smaller the GPU memory usage, but the slower the calculation speed. Defaults to 1.

    Returns:
        Union[np.ndarray, torch.Tensor]: Distance matrix of two vectors with shape n x m.
    """
    nx = ot.backend.get_backend(X_A, X_B)
    data_on_gpu = False
    if nx_torch(nx):
        if X_A.is_cuda:
            data_on_gpu = True
    type_as = X_A[0, 0].cpu() if nx_torch(nx) else X_A[0, 0]
    use_gpu = True if use_gpu and nx_torch(nx) and torch.cuda.is_available() else False
    chunk_flag = False
    while True:
        try:
            if chunk_num == 1:
                DistMat = _dist(X_A, X_B, "euc")
                break
            else:
                # convert to numpy to save the GPU memory
                if chunk_flag == False:
                    X_A, X_B = nx.to_numpy(X_A), nx.to_numpy(X_B)
                chunk_flag = True
                # chunk
                X_As = np.array_split(X_A, chunk_num, axis=0)
                X_Bs = np.array_split(X_B, chunk_num, axis=0)
                arr = []  # array for temporary storage of results
                for x_As in X_As:
                    arr2 = []  # array for temporary storage of results
                    for x_Bs in X_Bs:
                        if use_gpu:
                            arr2.append(
                                ot.dist(
                                    nx.from_numpy(x_As, type_as=type_as).cuda(),
                                    nx.from_numpy(x_Bs, type_as=type_as).cuda(),
                                ).cpu()
                            )
                        else:
                            arr2.append(
                                ot.dist(
                                    nx.from_numpy(x_As, type_as=type_as),
                                    nx.from_numpy(x_Bs, type_as=type_as),
                                )
                            )
                    arr.append(nx.concatenate(arr2, axis=1))
                DistMat = nx.concatenate(arr, axis=0)  # not convert to GPU
                break
        except:
            chunk_num = chunk_num * 2
            print("dist chunk more")
    if data_on_gpu and chunk_num != 1 and return_gpu:
        DistMat = DistMat.cuda()
    return DistMat


def voxel_data(
    coords: Union[np.ndarray, torch.Tensor],
    gene_exp: Union[np.ndarray, torch.Tensor],
    voxel_size: Optional[float] = None,
    voxel_num: Optional[int] = 10000,
):
    """
    Voxelization of the data.
    Parameters
    ----------
    coords: np.ndarray or torch.Tensor
        The coordinates of the data points.
    gene_exp: np.ndarray or torch.Tensor
        The gene expression of the data points.
    voxel_size: float
        The size of the voxel.
    voxel_num: int
        The number of voxels.
    Returns
    -------
    voxel_coords: np.ndarray or torch.Tensor
        The coordinates of the voxels.
    voxel_gene_exp: np.ndarray or torch.Tensor
        The gene expression of the voxels.
    """
    nx = ot.backend.get_backend(coords, gene_exp)
    N, D = coords.shape[0], coords.shape[1]
    coords = nx.to_numpy(coords)
    gene_exp = nx.to_numpy(gene_exp)

    # create the voxel grid
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)
    if voxel_size is None:
        voxel_size = np.sqrt(np.prod(max_coords - min_coords)) / (np.sqrt(N) / 5)
        # print(voxel_size)
    voxel_steps = (max_coords - min_coords) / int(np.sqrt(voxel_num))
    voxel_coords = [
        np.arange(min_coord, max_coord, voxel_step)
        for min_coord, max_coord, voxel_step in zip(min_coords, max_coords, voxel_steps)
    ]
    voxel_coords = np.stack(np.meshgrid(*voxel_coords), axis=-1).reshape(-1, D)
    voxel_gene_exps = np.zeros((voxel_coords.shape[0], gene_exp.shape[1]))
    is_voxels = np.zeros((voxel_coords.shape[0],))
    # assign the data points to the voxels
    for i, voxel_coord in enumerate(voxel_coords):
        dists = np.sqrt(np.sum((coords - voxel_coord) ** 2, axis=1))
        mask = dists < voxel_size / 2
        if np.any(mask):
            voxel_gene_exps[i] = np.mean(gene_exp[mask], axis=0)
            is_voxels[i] = 1
    voxel_coords = voxel_coords[is_voxels == 1, :]
    voxel_gene_exps = voxel_gene_exps[is_voxels == 1, :]
    return voxel_coords, voxel_gene_exps


def inlier_from_NN(
    train_x,
    train_y,
    distance,
):
    N, D = train_x.shape[0], train_x.shape[1]
    alpha = 1
    distance = np.maximum(0, distance)
    normalize = np.max(distance) / (np.log(10) * 2)
    distance = distance / (normalize)
    R = np.eye(D)
    t = np.ones((D, 1))
    y_hat = train_x
    sigma2 = np.sum((y_hat - train_y) ** 2) / (D * N)
    weight = np.exp(-distance * alpha)
    init_weight = weight
    P = np.multiply(np.ones((N, 1)), weight)
    max_iter = 100
    alpha_end = 0.1
    alpha_decrease = np.power(alpha_end / alpha, 1 / (max_iter - 20))
    gamma = 0.5
    a = np.maximum(
        np.prod(np.max(train_x, axis=0) - np.min(train_x, axis=0)),
        np.prod(np.max(train_y, axis=0) - np.min(train_y, axis=0)),
    )
    Sp = np.sum(P)
    for iter in range(max_iter):
        # solve rigid transformation
        mu_x = np.sum(np.multiply(train_x, P), 0) / (Sp)
        mu_y = np.sum(np.multiply(train_y, P), 0) / (Sp)

        X_mu, Y_mu = train_x - mu_x, train_y - mu_y
        A = np.dot(Y_mu.T, np.multiply(X_mu, P))
        svdU, svdS, svdV = np.linalg.svd(A)
        C = np.eye(D)
        C[-1, -1] = np.linalg.det(np.dot(svdU, svdV))
        R = np.dot(np.dot(svdU, C), svdV)
        t = mu_y - np.dot(mu_x, R.T)
        y_hat = np.dot(train_x, R.T) + t
        # get P
        term1 = np.multiply(
            np.exp(-(np.sum((train_y - y_hat) ** 2, 1, keepdims=True)) / (2 * sigma2)),
            weight,
        )
        outlier_part = (
            np.max(weight)
            * (1 - gamma)
            * np.power((2 * np.pi * sigma2), D / 2)
            / (gamma * a)
        )
        P = term1 / (term1 + outlier_part)
        Sp = np.sum(P)
        gamma = np.minimum(np.maximum(Sp / N, 0.01), 0.99)
        P = np.maximum(P, 1e-6)

        # update sigma2
        sigma2 = np.sum(np.multiply((y_hat - train_y) ** 2, P)) / (D * Sp)
        if iter > 20:
            alpha = alpha * alpha_decrease
            weight = np.exp(-distance * alpha)
            weight = weight / np.max(weight)

    fix_sigma2 = 1e-2
    fix_gamma = 0.1
    term1 = np.multiply(
        np.exp(-(np.sum((train_y - y_hat) ** 2, 1, keepdims=True)) / (2 * fix_sigma2)),
        weight,
    )
    outlier_part = (
        np.max(weight)
        * (1 - fix_gamma)
        * np.power((2 * np.pi * fix_sigma2), D / 2)
        / (fix_gamma * a)
    )
    P = term1 / (term1 + outlier_part)
    Sp = np.sum(P)
    gamma = np.minimum(np.maximum(Sp / N, 0.01), 0.99)
    return P, R, t, init_weight, sigma2, gamma


def coarse_rigid_alignment(
    coordsA: Union[np.ndarray, torch.Tensor],
    coordsB: Union[np.ndarray, torch.Tensor],
    X_A: Union[np.ndarray, torch.Tensor],
    X_B: Union[np.ndarray, torch.Tensor],
    dissimilarity: str = "kl",
    top_K: int = 10,
    transformed_points: Optional[Union[np.ndarray, torch.Tensor]] = None,
):
    nx = ot.backend.get_backend(coordsA, coordsB)
    if transformed_points is None:
        transformed_points = coordsA
    N, M, D = coordsA.shape[0], coordsB.shape[0], coordsA.shape[1]

    coordsA, X_A = voxel_data(
        coords=coordsA,
        gene_exp=X_A,
        voxel_num=max(min(int(N / 20), 1000), 100),
    )
    coordsB, X_B = voxel_data(
        coords=coordsB,
        gene_exp=X_B,
        voxel_num=max(min(int(M / 20), 1000), 100),
    )

    DistMat = calc_exp_dissimilarity(X_A=X_A, X_B=X_B, dissimilarity=dissimilarity)

    transformed_points = nx.to_numpy(transformed_points)
    sub_coordsA = coordsA

    item2 = np.argpartition(DistMat, top_K, axis=0)[:top_K, :].T
    item1 = np.repeat(np.arange(DistMat.shape[1])[:, None], top_K, axis=1)
    NN1 = np.dstack((item1, item2)).reshape((-1, 2))
    distance1 = DistMat.T[NN1[:, 0], NN1[:, 1]]

    ## construct nearest neighbor set using brute force
    item1 = np.argpartition(DistMat, top_K, axis=1)[:, :top_K]
    item2 = np.repeat(np.arange(DistMat.shape[0])[:, None], top_K, axis=1)
    NN2 = np.dstack((item1, item2)).reshape((-1, 2))
    distance2 = DistMat.T[NN2[:, 0], NN2[:, 1]]

    NN = np.vstack((NN1, NN2))
    distance = np.r_[distance1, distance2]

    train_x, train_y = sub_coordsA[NN[:, 1], :], coordsB[NN[:, 0], :]

    # print(train_x)
    R_flip = np.eye(D)
    R_flip[-1, -1] = -1

    P, R, t, init_weight, sigma2, gamma = inlier_from_NN(
        train_x, train_y, distance[:, None]
    )

    P2, R2, t2, init_weight, sigma2_2, gamma_2 = inlier_from_NN(
        np.dot(train_x, R_flip), train_y, distance[:, None]
    )

    if gamma_2 > (1.5 * gamma):
        P = P2
        R = R2
        t = t2
        sigma2 = sigma2_2
        R = np.dot(R, R_flip)
        print("flip")
    inlier_threshold = min(P[np.argsort(-P[:, 0])[20], 0], 0.5)
    inlier_set = np.where(P[:, 0] > inlier_threshold)[0]
    inlier_x, inlier_y = train_x[inlier_set, :], train_y[inlier_set, :]
    inlier_P = P[inlier_set, :]

    transformed_points = np.dot(transformed_points, R.T) + t
    inlier_x = np.dot(inlier_x, R.T) + t
    return transformed_points, inlier_x, inlier_y, inlier_P, R, t


def get_optimal_R(
    coordsA: Union[np.ndarray, torch.Tensor],
    coordsB: Union[np.ndarray, torch.Tensor],
    P: Union[np.ndarray, torch.Tensor],
    R_init: Union[np.ndarray, torch.Tensor],
):
    """Get the optimal rotation matrix R

    Args:
        coordsA (Union[np.ndarray, torch.Tensor]): The first input matrix with shape n x d
        coordsB (Union[np.ndarray, torch.Tensor]): The second input matrix with shape n x d
        P (Union[np.ndarray, torch.Tensor]): The optimal transport matrix with shape n x n

    Returns:
        Union[np.ndarray, torch.Tensor]: The optimal rotation matrix R with shape d x d
    """
    nx = ot.backend.get_backend(coordsA, coordsB, P, R_init)
    NA, NB, D = coordsA.shape[0], coordsB.shape[0], coordsA.shape[1]
    used_gpu = False
    if nx_torch(nx) and coordsA.is_cuda:
        coordsA = coordsA.cpu()
        coordsB = coordsB.cpu()
        R_init = R_init.cpu()
        used_gpu = True
    if nx_torch(nx) and P.is_cuda:
        P = P.cpu()
    Sp = nx.einsum("ij->", P)
    K_NA = nx.einsum("ij->i", P)
    K_NB = nx.einsum("ij->j", P)
    VnA = nx.zeros(coordsA.shape, type_as=coordsA[0, 0])
    mu_XnA, mu_VnA, mu_XnB = (
        _dot(nx)(K_NA, coordsA) / Sp,
        _dot(nx)(K_NA, VnA) / Sp,
        _dot(nx)(K_NB, coordsB) / Sp,
    )
    XnABar, VnABar, XnBBar = coordsA - mu_XnA, VnA - mu_VnA, coordsB - mu_XnB
    A = -_dot(nx)(nx.einsum("ij,i->ij", VnABar, K_NA).T - _dot(nx)(P, XnBBar).T, XnABar)

    svdU, svdS, svdV = _linalg(nx).svd(A)
    C = _identity(nx, D, type_as=coordsA[0, 0])
    C[-1, -1] = _linalg(nx).det(_dot(nx)(svdU, svdV))
    R = _dot(nx)(_dot(nx)(svdU, C), svdV)
    t = mu_XnB - mu_VnA - _dot(nx)(mu_XnA, R.T)
    optimal_RnA = _dot(nx)(coordsA, R.T) + t
    if used_gpu:
        return optimal_RnA.cuda(), R.cuda(), t.cuda()
    else:
        return optimal_RnA, R, t


#################################
# Funcs between Numpy and Torch #
#################################


# Empty cache
def empty_cache(device: str = "cpu"):
    if device != "cpu":
        torch.cuda.empty_cache()


# Check if nx is a torch backend
nx_torch = lambda nx: True if isinstance(nx, ot.backend.TorchBackend) else False

# Concatenate expression matrices
_cat = (
    lambda nx, x, dim: torch.cat(x, dim=dim)
    if nx_torch(nx)
    else np.concatenate(x, axis=dim)
)
_unique = (
    lambda nx, x, dim: torch.unique(x, dim=dim)
    if nx_torch(nx)
    else np.unique(x, axis=dim)
)
_var = lambda nx, x, dim: torch.var(x, dim=dim) if nx_torch(nx) else np.var(x, axis=dim)

_data = (
    lambda nx, data, type_as: torch.tensor(
        data, device=type_as.device, dtype=type_as.dtype
    )
    if nx_torch(nx)
    else np.asarray(data, dtype=type_as.dtype)
)
_unsqueeze = lambda nx: torch.unsqueeze if nx_torch(nx) else np.expand_dims
_mul = lambda nx: torch.multiply if nx_torch(nx) else np.multiply
_power = lambda nx: torch.pow if nx_torch(nx) else np.power
_psi = lambda nx: torch.special.psi if nx_torch(nx) else psi
_pinv = lambda nx: torch.linalg.pinv if nx_torch(nx) else pinv
_dot = lambda nx: torch.matmul if nx_torch(nx) else np.dot
_identity = (
    lambda nx, N, type_as: torch.eye(N, dtype=type_as.dtype, device=type_as.device)
    if nx_torch(nx)
    else np.identity(N, dtype=type_as.dtype)
)
_linalg = lambda nx: torch.linalg if nx_torch(nx) else np.linalg
_prod = lambda nx: torch.prod if nx_torch(nx) else np.prod
_pi = lambda nx: torch.pi if nx_torch(nx) else np.pi
_chunk = (
    lambda nx, x, chunk_num, dim: torch.chunk(x, chunk_num, dim=dim)
    if nx_torch(nx)
    else np.array_split(x, chunk_num, axis=dim)
)
_randperm = lambda nx: torch.randperm if nx_torch(nx) else np.random.permutation
_roll = lambda nx: torch.roll if nx_torch(nx) else np.roll
_choice = (
    lambda nx, length, size: torch.randperm(length)[:size]
    if nx_torch(nx)
    else np.random.choice(length, size, replace=False)
)
_topk = (
    lambda nx, x, topk, axis: torch.topk(x, topk, dim=axis)[1]
    if nx_torch(nx)
    else np.argpartition(x, topk, axis=axis)
)
_dstack = lambda nx: torch.dstack if nx_torch(nx) else np.dstack
_vstack = lambda nx: torch.vstack if nx_torch(nx) else np.vstack
_hstack = lambda nx: torch.hstack if nx_torch(nx) else np.hstack
