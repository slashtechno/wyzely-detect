FROM python:3.10-bullseye

# Install Dlib (for face_recognition)
RUN apt-get -y update && apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    software-properties-common \
    zip
RUN apt-get clean
RUN rm -rf /tmp/* /var/tmp/*
ENV CFLAGS=-static
# Upgrade pip
RUN pip3 install --upgrade pip
# Copy directory to container
WORKDIR /app
COPY . ./
# Install from requirements.txt
RUN pip3 install -r requirements.txt
ENTRYPOINT ["python", "main.py"]