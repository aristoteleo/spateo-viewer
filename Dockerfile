FROM kitware/trame

COPY --chown=trame-user:trame-user . /deploy
RUN sudo apt-get install libx11-dev
RUN export TRAME_BUILD_ONLY=1 && /opt/trame/entrypoint.sh build
