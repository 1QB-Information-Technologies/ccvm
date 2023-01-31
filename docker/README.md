# Docker

## Images

#### Base Image

The base dependency image used for `ccvm` can be found on [Docker Hub](https://hub.docker.com/r/1qbit/ubuntu20-python/tags).

Tags:
- `3.10`: Base Ubuntu 20.04 w/ Python + LaTex (~2 GB)
- `3.10-slim`: Base Ubuntu 20.04 w/ Python (~200 MB)

### Pre-built Image (1QBit)

- `quay.io/1qbit/ccvm:latest` - Full `ccvm` image
- `quay.io/1qbit/ccvm:slim` - Excludes `texlive` (required for LaTex and plot generation in `ccvmplotlib`)

## Containers

### `ccvm`

The `ccvm` container includes all required dependencies.

### `ccvm-slim`

The `ccvm-slim` container includes everything but the `texlive` package that allows for LaTex plot labelling, which means you can still use the solvers but won't be able to plot with `ccvmplotlib`.

