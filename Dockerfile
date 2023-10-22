FROM python:3.10.5-buster

RUN apt update && apt install libgl1 -y
RUN pip install poetry

WORKDIR /app

COPY . .

RUN poetry install

ENTRYPOINT ["poetry", "run", "python", "-m", "wyzely_detect"]