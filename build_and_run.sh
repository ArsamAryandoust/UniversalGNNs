docker build Docker -t ubuntu-torch
docker run -it --gpus 1 -v ~/UniversalGNNs/:/UniversalGNNs/ -v ~/TasksEnergyTransition/:/TasksEnergyTransition/ ubuntu-torch
# docker run -it --gpus 1 -v ~/UniversalGNNs/:/UniversalGNNs/ -v /home/shared_cp/TasksEnergyTransition/:/TasksEnergyTransition/ ubuntu-torch