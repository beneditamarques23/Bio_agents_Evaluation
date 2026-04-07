FROM python:3.11-slim

ENV POETRY_VERSION 1.7.0 \
    PYTHONDONTWRITEBYTECODE 1 \
    PYTHONUNBUFFERED 1 \
    POETRY_NO_INTERACTION 1 \

RUN pip install poetry==$POETRY_VERSION

RUN poetry config virtualenvs.create false

WORKDIR /app

COPY pyproject.toml poetry.lock README.md ./

RUN poetry install --no-interaction --without dev -vvv

COPY src ./src/

CMD ["poetry", "run", "python", "src/main.py"]
