FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Default: run the batch pipeline once (matches the scheduled-batch
# architecture decision -- this image is invoked by the GitHub Actions
# workflow, not left running as a service).
CMD ["python", "run_pipeline.py"]

# To run the RQ worker process instead (for future live-enqueued jobs,
# distinct from the scheduled batch run above), override the command:
#   docker run <image> python worker.py
