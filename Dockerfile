FROM python:3.10.5-buster

LABEL org.opencontainers.image.description "Docker image for running wyzely-detect"
LABEL org.opencontainers.image.source "https://github.com/slashtechno/wyzely-detect"

RUN apt update && apt install libgl1 -y
RUN pip install poetry


WORKDIR /app

COPY . .

RUN poetry install

RUN poetry run pip uninstall -y torchvision 
RUN poetry run pip install torchvision 

ENTRYPOINT ["poetry", "run", "python", "-m", "--", "wyzely_detect", "--no-display"]