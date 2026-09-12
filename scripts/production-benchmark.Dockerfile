FROM python:3.11-slim-bookworm
ARG ACTIVITYSIM_COMMIT
ARG SHARROW_COMMIT
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*
# Fetch exact objects with ancestry/tags so setuptools-scm derives real versions.
# Verify the checked-out object before installing either package.
RUN git init /opt/activitysim && cd /opt/activitysim \
    && git remote add origin https://github.com/ActivitySim/ActivitySim.git \
    && git fetch --tags origin "$ACTIVITYSIM_COMMIT" && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "$ACTIVITYSIM_COMMIT"
RUN git init /opt/sharrow && cd /opt/sharrow \
    && git remote add origin https://github.com/ActivitySim/Sharrow.git \
    && git fetch --tags origin "$SHARROW_COMMIT" && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "$SHARROW_COMMIT"
RUN python -m pip install --no-cache-dir /opt/activitysim /opt/sharrow \
    'multimethod<2' 'pandas<3' pyyaml pyarrow \
    && python -m pip check && python -m pip freeze > /opt/pip-freeze.txt
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONHASHSEED=0 DASK_SCHEDULER=synchronous
WORKDIR /model
ENTRYPOINT ["python", "/benchmark/worker.py"]
