FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app
COPY src/inference.py src/inference.py
COPY src/model.py src/model.py
COPY checkpoints/full_v2/best_model.pt checkpoints/full_v2/best_model.pt

ENTRYPOINT ["python", "src/inference.py"]
