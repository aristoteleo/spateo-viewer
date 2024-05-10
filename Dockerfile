FROM kitware/trame

COPY --chown=trame-user:trame-user . /deploy

ENV TRAME_CLIENT_TYPE=vue2
RUN export TRAME_BUILD_ONLY=1 && /opt/trame/entrypoint.sh build
