FROM kitware/trame

COPY --chown=trame-user:trame-user . /deploy

ENV TRAME_CLIENT_TYPE=vue2
RUN apt-get update && apt-get install -y libx11-dev libgl1-mesa-glx libglib2.0-0 libxrender1
RUN export TRAME_BUILD_ONLY=1 && /opt/trame/entrypoint.sh build
