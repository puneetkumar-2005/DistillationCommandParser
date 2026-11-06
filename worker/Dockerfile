FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY worker.py .
# No model weights in this image. The prompt arrives via a ConfigMap
# mounted at /prompts/teacher_prompt.txt (see worker/jobs.yaml).
CMD ["python", "worker.py"]
