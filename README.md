<p align="center">
  <img height="150" src="https://github.com/aristoteleo/spateo-viewer/blob/main/stviewer/assets/image/spateo_logo.png" />
</p>

## Spateo-viewer: the "Google earth" browser of spatial transcriptomics

[**Spateo-viewer**](https://github.com/aristoteleo/spateo-viewer) is the “Google earth” of spatial transcriptomics. 
Relying on a set of powerful libraries and tools in the Python ecosystem, such as Trame, PyVista, VTK, etc., it delivers 
a complete web application solution of creating convenient, vivid, and lightweight interface for 3D reconstruction and 
visualization of [**Spateo**](https://github.com/aristoteleo/spateo-release) downstream analysis results. Currently, 
Spateo-viewer includes two major applications, **interactive-viewer** and **static-viewer**, which are respectively 
dedicated to the 3D reconstruction of spatial transcriptomics and the visualization of spatial transcriptomics analysis results.

Please view the [**Slides**](https://github.com/aristoteleo/spateo-viewer/blob/main/usage/spateo-viewer.html) to learn more about Spateo-viewer.

## Highlights

* In the interactive-viewer, 3D serial slices of spatial transcriptomics datasets can be aligned to create 3D models. The 3D model can be also cleaned up by freely clipping and editing. 
* In the static-viewer, users can not only visualize gene expression, but also easily switch between raw and different types of normalized data or data layers. Users can also visualize all cell annotation information such as cell size, cell type, tissue type, etc. All done in 3D space!
* Static-viewer allows users to not only visualize the point cloud model and mesh model of the whole embryo, but also for individual organ or tissue type at the same time. It even visualizes morphogenesis vector field model to animate how cell move in the physical 3D space. 
* Spateo-viewer can not only run on the local computer, but also run freely on the remote server. 
* Users can upload custom files in the web application, or access to custom files in local folders when running Spateo-Viewer as a stand alone App.(See [**StaticViewerUsage**](https://github.com/aristoteleo/spateo-viewer/blob/main/usage/StaticViewerUsage.md) or [**InteractiveViewerUsage**](https://github.com/aristoteleo/spateo-viewer/blob/main/usage/InteractiveViewerUsage.md))

## Installation

You can clone the [**Spateo-viewer**](https://github.com/aristoteleo/spateo-viewer) with ``git`` and install dependencies with ``pip``:

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
