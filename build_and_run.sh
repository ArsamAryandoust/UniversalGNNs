docker build Docker -t ubuntu-torch
docker run -it --ipc=host --gpus "device=0" -v ~/UniversalGNNs/:/UniversalGNNs/ -v ~/EnergyTransitionTasks/:/EnergyTransitionTasks/ ubuntu-torch

