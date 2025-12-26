FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

RUN apt-get clean && rm -rf /var/lib/apt/lists/*
RUN apt-get -o Acquire::ForceIPv4=true update
RUN apt-get install -y python3.10 python3-pip git curl ffmpeg
RUN pip3 install --upgrade pip

WORKDIR /app

COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN mkdir -p model

COPY whisper_daemon.py .

CMD ["python3", "whisper_daemon.py"]
