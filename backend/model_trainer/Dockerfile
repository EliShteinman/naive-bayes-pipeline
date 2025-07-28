FROM python:3.11-slim AS builder
# Change 1: Set the working directory to /code to avoid name conflicts
WORKDIR /code

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY model_trainer/requirements.txt .

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim AS final
# Change 1: Set the working directory to /code to avoid name conflicts
WORKDIR /code

# Change 2: Update paths in the RUN command to match the new WORKDIR
RUN addgroup --system app && \
    adduser --system --ingroup app app && \
    mkdir -p /code/app && \
    mkdir -p /code/data && \
    chown -R app:app /code

USER app

COPY --from=builder /opt/venv /opt/venv

COPY data/mushroom_decoded.csv ./data/

# Your explicit file copies - UNCHANGED AND CORRECT
COPY model_trainer/__init__.py ./app/
COPY app/data_cleaner.py ./app/
COPY app/data_handler.py ./app/
COPY app/data_splitter.py ./app/
COPY app/logger_config.py ./app/
COPY app/model_evaluator.py ./app/
COPY app/naive_bayes_model_builder.py ./app/
COPY app/model_artifact.py ./app/

# Your explicit file copies - UNCHANGED AND CORRECT
COPY model_trainer/__init__.py ./
COPY model_trainer/config.py ./
COPY model_trainer/main.py ./
COPY model_trainer/application_manager.py ./

# Change 3: Add PYTHONPATH so Python can find the 'app' package from the root WORKDIR
ENV PYTHONPATH=/code \
    PATH="/opt/venv/bin:$PATH" \
    LOG_TO_FILE=false

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]