ARG PYTORCH_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

WORKDIR /workspace/tridim

COPY requirements-runtime.txt .
RUN python -m pip install --no-cache-dir -r requirements-runtime.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV TRIDIM_DATA_ROOT=/datasets

CMD ["python", "scripts/smoke_test_models.py"]
