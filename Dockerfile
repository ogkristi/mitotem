FROM ghcr.io/prefix-dev/pixi:0.34.0-bookworm-slim AS build

WORKDIR /app
COPY pyproject.toml pixi.lock ./
RUN mkdir mitotem
RUN touch mitotem/__init__.py
RUN pixi install --locked

FROM gcr.io/distroless/base-debian12 AS production

WORKDIR /app
COPY --from=build /app/.pixi/envs/default /app/.pixi/envs/default
COPY . .

# from pixi shell-hook
ENV PATH=/app/.pixi/envs/default/bin:$PATH
ENV CONDA_PREFIX=/app/.pixi/envs/default

ENTRYPOINT [ "python", "mitotem/main.py", "predict", "/input", "-o", "/output" ]