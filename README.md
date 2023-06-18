<p align="center">
  <img height="150" src="https://github.com/aristoteleo/spateo-viewer/blob/main/stviewer/assets/image/spateo_logo.png" />
</p>

## Spateo-Viewer: Web application demonstrating 3D modeling of spatial transcriptomics

[**Spateo-Viewer**](https://github.com/aristoteleo/spateo-viewer) uses existing libraries and tools (such as Trame, 
PyVista, VTK, etc.) to create convenient, vivid, and lightweight content for the visualization of [**Spateo**](https://github.com/aristoteleo/spateo-release)
downstream analysis results. Currently, [**Spateo-Viewer**](https://github.com/aristoteleo/spateo-viewer) includes two 
web applications, **static-viewer** and **interactive-viewer**, which are respectively dedicated to the visualization of 3D 
spatial transcriptomics analysis results and the region clipping of spatial transcriptomics.

![Spateo-viewer](https://github.com/aristoteleo/spateo-viewer/blob/main/stviewer/assets/image/spateo-viewer.pdf)

## Highlights

* In static-viewer, users can not only visualize all genes, but also change the gene expression profile by changing the matrix. Additionally, users can also visualize all the information contained in ``anndata.obs``, such as cell size, cell type, tissue type, etc.
* In static-viewer, users can not only visualize the point cloud model and mesh model of the complete spatial transcriptomics data, but also visualize the point cloud model and mesh model of each tissue type at the same time, and even visualize the vector field model and trajectory model of cell migration fate in some tissues.
* In interactive-viewer, users can freely clip and save the clipped spatial transcriptomics model.
* Users can upload custom files in the web application, or add the absolute path of the custom files in assets to open the custom models in Spateo-Viewer.(See [**StaticViewerUsage**](https://github.com/aristoteleo/spateo-viewer/blob/main/usage/StaticViewerUsage.md) or [**InteractiveViewerUsage**](https://github.com/aristoteleo/spateo-viewer/blob/main/usage/InteractiveViewerUsage.md))

## Installation

You can clone the [**Spateo-Viewer**](https://github.com/aristoteleo/spateo-viewer) with ``git`` and install dependencies with ``pip``:

    git clone https://github.com/aristoteleo/spateo-viewer.git
    cd spateo-viewer
    pip install -r requirements.txt

## Usage

#### Run the static-viewer application:

    python stv_static_app.py --port 1234

See the [**StaticViewerUsage**](https://github.com/aristoteleo/spateo-viewer/blob/main/usage/StaticViewerUsage.md) for more details.

#### Run the interactive-viewer application:

    python stv_interactive_app.py --port 1234

See the [**InteractiveViewerUsage**](https://github.com/aristoteleo/spateo-viewer/blob/main/usage/InteractiveViewerUsage.md) for more details.

## Citation

Xiaojie Qiu1$\*, Daniel Y. Zhu3$, Jiajun Yao2, 4, 5, 6$, Zehua Jing2, 4,7$, Lulu Zuo8$, Mingyue Wang2, 4, 9, 10$, Kyung
Hoi (Joseph) Min11, Hailin Pan2, 4, Shuai Wang2, 4, 7, Sha Liao4, Yiwei Lai4, Shijie Hao2, 4, 7, Yuancheng Ryan Lu1, 
Matthew Hill17, Jorge D. Martin-Rufino17, Chen Weng1, Anna Maria Riera-Escandell18, Mengnan Chen2, 4, Liang Wu4, Yong 
Zhang4, Xiaoyu Wei2, 4, Mei Li4, Xin Huang4, Rong Xiang2, 4, 7, Zhuoxuan Yang4, 12, Chao Liu4, Tianyi Xia4, Yingxin 
Liang10, Junqiang Xu4,7, Qinan Hu9, 10, Yuhui Hu9, 10, Hongmei Zhu8, Yuxiang Li4, Ao Chen4, Miguel A. Esteban4, Ying 
Gu2, 4,7, Douglas A. Lauffenburger3, Xun Xu2, 4, 13, Longqi Liu2, 4, 14, 15\*, Jonathan S. Weissman1,19, 20\*, Shiping 
Liu2, 4, 14, 15, 16\*, Yinqi Bai2, 4\*  $Co-first authors; *:Corresponding authors
 
[Spateo: multidimensional spatiotemporal modeling of single-cell spatial transcriptomics](https://www.biorxiv.org/content/10.1101/2022.12.07.519417v1)
