FROM 1qbit/ubuntu20-python:3.10
ARG HOME_DIR=/workspace
ENV PYTHONPATH="${PYTHONPATH}:${HOME_DIR}"

WORKDIR ${HOME_DIR}

COPY . .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt
