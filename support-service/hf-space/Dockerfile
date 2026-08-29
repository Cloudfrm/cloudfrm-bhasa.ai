FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

COPY --chown=user . /home/user/app

EXPOSE 7860

CMD ["python", "app.py"]
