# Best CUDA Project Ever

Copies an array to the device and assigns a unique calculated value to each entry depending on which thread it ran on.

## Setup

1. Install the NVIDIA toolkit!

2. `mkdir build`

## Build
```sh
make -Wno-deprecated-gpu-targets
```
## Run
```sh
./build/program
```