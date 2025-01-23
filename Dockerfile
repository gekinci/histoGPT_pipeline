# syntax=docker/dockerfile:1
FROM python:3.10-slim

# install app dependencies and tools
RUN apt update && apt install -y git libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg

# cloning the repository
RUN git clone https://github.com/gekinci/histoGPT_pipeline.git

# Set the working directory
WORKDIR /histoGPT_pipeline

# Create and activate a virtual environment, and install Python dependencies
RUN pip install -r requirements.txt

EXPOSE 3000

CMD ["dagit", "-f", "ml_pipeline/main.py"]
