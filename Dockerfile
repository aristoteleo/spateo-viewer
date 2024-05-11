FROM kitware/trame

COPY --chown=trame-user:trame-user . /deploy
RUN apt-get update && apt-get install -y libx11-dev
RUN export TRAME_BUILD_ONLY=1 && /opt/trame/entrypoint.sh build
