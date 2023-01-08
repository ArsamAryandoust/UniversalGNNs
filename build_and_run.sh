docker build Docker -t ubuntu-torch
docker run -it --ipc=host --gpus 1 -v ~/Documents/projects/UniversalGNNs/:/UniversalGNNs/ -v ~/Documents/projects/TasksEnergyTransition/:/TasksEnergyTransition/ ubuntu-torch

# docker run -it --gpus 1 -v ~/UniversalGNNs/:/UniversalGNNs/ -v /home/shared_cp/TasksEnergyTransition/:/TasksEnergyTransition/ ubuntu-torch
