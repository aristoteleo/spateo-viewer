FROM kitware/trame

COPY --chown=trame-user:trame-user . /deploy

ENV TRAME_CLIENT_TYPE=vue2
RUN apt-get update && apt-get install -y libx11-dev libgl1-mesa-glx libglib2.0-0
RUN export TRAME_BUILD_ONLY=1 && /opt/trame/entrypoint.sh build
RUN . /deploy/server/venv/bin/activate && pip uninstall vtk -y && pip install --extra-index-url https://wheels.vtk.org vtk-osmesa

RUN mkdir -p /deploy/stviewer/assets/dataset/mouse_E115/h5ad
RUN mkdir -p /deploy/stviewer/assets/dataset/mouse_E115/matrices
RUN mkdir -p /deploy/stviewer/assets/dataset/mouse_E115/mesh_models
RUN mkdir -p /deploy/stviewer/assets/dataset/mouse_E115/pc_models
ADD https://www.dropbox.com/scl/fi/8xs13xxhlilxhsbvzzlcf/mouse_E115.h5ad?rlkey=z4h6vyxhzwzinmv6hngiaxmz7&st=znamo2ap&dl=1 /deploy/stviewer/assets/dataset/mouse_E115/h5ad/mouse_E115.h5ad
ADD https://www.dropbox.com/scl/fi/rhxmwyh2yt5jo1bnmwh8g/X_sparse_martrix.npz?rlkey=m5ogihuul5fgpfh26uo9wcjc5&st=ke47exyf&dl=1 /deploy/stviewer/assets/dataset/mouse_E115/matrices/X_sparse_martrix.npz
ADD https://www.dropbox.com/scl/fi/gdx5teuml2xr09vm0x5zb/0_Embryo_mouse_E115_mesh_model.vtk?rlkey=zk8b7b86cfvmux0rqksg9s9k2&st=k6pluhh6&dl=1 /deploy/stviewer/assets/dataset/mouse_E115/mesh_models/0_Embryo_mouse_E115_mesh_model.vtk
ADD https://www.dropbox.com/scl/fi/950z5zefkzc8mrea90gja/0_Embryo_mouse_E115_pc_model.vtk?rlkey=jv5kpl5oa26dcu5164f4r0f4j&st=35joa9e0&dl=1 /deploy/stviewer/assets/dataset/mouse_E115/pc_models/0_Embryo_mouse_E115_pc_model.vtk

RUN mkdir -p /deploy/stviewer/assets/dataset/mouse_E95/h5ad
RUN mkdir -p /deploy/stviewer/assets/dataset/mouse_E95/matrices
RUN mkdir -p /deploy/stviewer/assets/dataset/mouse_E95/mesh_models
RUN mkdir -p /deploy/stviewer/assets/dataset/mouse_E95/pc_models
ADD https://www.dropbox.com/scl/fi/uxmachlvh726kntmk6rqn/mouse_E95.h5ad?rlkey=4xi940x1uyllnilj0up76mx11&st=x8ybj4xa&dl=1 /deploy/stviewer/assets/dataset/mouse_E95/h5ad/mouse_E95.h5ad
ADD https://www.dropbox.com/scl/fi/qlvvyt5uy0ywt2gfy7y59/X_sparse_martrix.npz?rlkey=hh6pi1xh2koqjnmubpx6lp22w&st=s26ut25l&dl=1 /deploy/stviewer/assets/dataset/mouse_E95/matrices/X_sparse_martrix.npz
ADD https://www.dropbox.com/scl/fi/ksetxjsz5tz20wlv5z2ou/0_Embryo_mouse_E95_mesh_model.vtk?rlkey=xhoj8j1mvxa80vx0xjcibtuwd&st=ges5bxlv&dl=1 /deploy/stviewer/assets/dataset/mouse_E95/mesh_models/0_Embryo_mouse_E95_mesh_model.vtk
ADD https://www.dropbox.com/scl/fi/t07hhnlmxpi6l6v2yrzsf/0_Embryo_mouse_E95_pc_model.vtk?rlkey=sp81k9oa0dn0iia1rrtsdx0pn&st=47hx1i8l&dl=1 /deploy/stviewer/assets/dataset/mouse_E95/pc_models/0_Embryo_mouse_E95_pc_model.vtk

RUN chown -R trame-user:trame-user /deploy