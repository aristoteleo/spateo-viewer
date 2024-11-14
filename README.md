<p align="center">
  <img height="150" src="https://github.com/aristoteleo/spateo-viewer/blob/main/stviewer/assets/image/spateo_logo.png" />
</p>

## Spateo-viewer: the "Google earth" browser of spatial transcriptomics

[**Spateo-viewer**](https://github.com/aristoteleo/spateo-viewer) is the “Google earth” of spatial transcriptomics. 
Relying on a set of powerful libraries and tools in the Python ecosystem, such as Trame, PyVista, VTK, etc., it delivers 
a complete web application solution of creating convenient, vivid, and lightweight interface for 3D reconstruction and 
visualization of [**Spateo**](https://github.com/aristoteleo/spateo-release) downstream analysis results. Currently, 
Spateo-viewer includes two major applications, ***Explorer*** and ***Reconstructor***, which are respectively 
dedicated to the 3D reconstruction of spatial transcriptomics and the visualization of spatial transcriptomics analysis results.

Please download and read the corresponding [**Slides**](https://github.com/aristoteleo/spateo-viewer/blob/main/usage/spateo-viewer.pdf) to learn more about Spateo-viewer.

## Highlights

* In the ***Reconstructor***, 3D serial slices of spatial transcriptomics datasets can be aligned to create 3D models. The 3D model can be also cleaned up by freely clipping and editing. 
* In the ***Explorer***, users can not only visualize gene expression, but also easily switch between raw and different types of normalized data or data layers. Users can also visualize all cell annotation information such as cell size, cell type, tissue type, etc. All done in 3D space!
* Static-viewer allows users to not only visualize the point cloud model and mesh model of the whole embryo, but also for individual organ or tissue type at the same time. It even visualizes morphogenesis vector field model to animate how cell move in the physical 3D space. 
* Spateo-viewer can not only run on the local computer, but also run freely on the remote server. 
* Users can upload custom files in the web application, or access to custom files in local folders when running Spateo-Viewer as a stand alone App.(See [**ExplorerUsage**](https://github.com/aristoteleo/spateo-viewer/blob/main/usage/ExplorerUsage.md) or [**ReconstructorUsage**](https://github.com/aristoteleo/spateo-viewer/blob/main/usage/ReconstructorUsage.md))

## Installation

You can clone the [**Spateo-viewer**](https://github.com/aristoteleo/spateo-viewer) with ``git`` and install dependencies with ``pip``:

    git clone https://github.com/aristoteleo/spateo-viewer.git
    cd spateo-viewer
    pip install -r requirements.txt

## Usage

#### Run the *Explorer* application:

    python stv_explorer.py --port 1234

See the [**ExplorerUsage**](https://github.com/aristoteleo/spateo-viewer/blob/main/usage/ExplorerUsage.md) for more details.

#### Run the *Reconstructor* application:

    python stv_reconstructor.py --port 1234

See the [**ReconstructorUsage**](https://github.com/aristoteleo/spateo-viewer/blob/main/usage/ReconstructorUsage.md) for more details.

## Sample Datasets

#### [**Mouse E9.5 dataset**](https://github.com/aristoteleo/spateo-viewer/tree/main/stviewer/assets/dataset/mouse_E95): 
- **h5ad/mouse_E95_demo.h5ad**：Single-cell resolution Stereo-seq data with alignment and cell annotation by the Spateo team. This data only contains 1000 highly variable genes. If you need the raw data, please check the [CNGB website](https://db.cngb.org/stomics/mosta/download/).
- **matrices**: Contains various gene expression matrices for .h5ad data.
- **mesh_models**：Contains mesh model of mouse embryo.
- **pc_models**：Contains point cloud model of mouse embryo.

#### [**Drosophila S11 dataset**](https://github.com/aristoteleo/spateo-viewer/tree/main/stviewer/assets/dataset/drosophila_S11):
- **h5ad/S11_cellbin_demo.h5ad**：Single-cell resolution Stereo-seq data with alignment and cell annotation by the Spateo team. This data only contains 1000 highly variable genes. If you need the raw data, please check the [CNGB website](https://db.cngb.org/stomics/mosta/download/).
- **mesh_models**：Contains mesh models of drosophila embryos and various organs.
- **pc_models**：Contains point cloud models of drosophila embryos and various organs.

## Citation

[<b> Spatiotemporal modeling of molecular holograms </b>](https://www.cell.com/cell/fulltext/S0092-8674(24)01159-0)

Xiaojie Qiu1, 7, 8\$\*, Daniel Y. Zhu3\$, Yifan Lu1, 7, 8, 9\$, Jiajun Yao2, 4, 10\$, Zehua Jing2, 4, 11\$, Kyung Hoi (Joseph) Min12\$, Mengnan Cheng2，6\$, Hailin Pan6, Lulu Zuo6, Samuel King13, Qi Fang2, 6, Huiwen Zheng2, 11, Mingyue Wang2, 14, Shuai Wang2, 11, Qingquan Zhang25, Sichao Yu5, Sha Liao6, 17, 18, Chao Liu15, Xinchao Wu2, 4, 16, Yiwei Lai6, Shijie Hao2, Zhewei Zhang2, 4, 16, Liang Wu18, Yong Zhang15, Mei Li17, Zhencheng Tu2, 11, Jinpei Lin2, 4, Zhuoxuan Yang2, 16, Yuxiang Li15, Ying Gu2, 6, 11, Ao Chen6, 17, 18, Longqi Liu2, 19, 20, Jonathan S. Weissman5, 22, 23, Jiayi Ma9*, Xun Xu2, 11, 21*, Shiping Liu2, 19, 20, 24*, Yinqi Bai4, 26*

$Co-first authors; *:Corresponding authors
