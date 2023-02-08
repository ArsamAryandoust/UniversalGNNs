<img src="https://img.shields.io/badge/experiments-2-blue"/>

# Universal graph neural networks for multi-task transfer learning

<img src="/figures/UniversalDataGraph.png" />

A fundamental research question that arises from the recent development of universal deep learning models like Gato or ChatGPT, is whether such models are also suitable for tackling important climate change related prediction problems. We design the UniversalGNN model for this with an emphasis on handling climate change related themes such as spatio-temporal data. We argue that our UniversalGNN is able to solve arbitrary prediction tasks using arbitrary non-iid data, and demonstrate the performance of a first simple implementation of it on three prediction tasks that are important for enhancing the global energy transition. We find that our current model has theoretical properties that can be improved and that its implementation is not yet able to benefit from transfer learning across different domains and tasks or yet perform consistently better than task-specific but simple multi-layer perceptrons. We propose multiple improvements to enhance the theoretical properties and implementation of our UniversalGNN, and to increase the general reliability and applicability of our experiments.


### Getting started
Download this repository and the `EnergyTransitionTasks` one to your home directory.
The `EnergyTransitionTasks` repository contains the datasets we use here:

```
cd 
git clone https://github.com/ArsamAryandoust/UniversalGNNs.git
git clone https://github.com/ArsamAryandoust/EnergyTransitionTasks.git
cd UniversalGNNs
```


### Docker
The easiest way to build and run the Docker container is with the `build_and_run.sh` script inside the `UniversalGNNs` folder.
To do this, execute the following command:

```
./build_and_run.sh
```


### Experiments
All the models (included the baselines) can be trained from the `main.py` file inside of the `src` folder. 
The easiest way to do so is to start the Docker container built as above and inside of it run:

```
python3 src/main.py --help
```

This will show the available command line arguments that control the models that will be trained. 
For example, training the UniversalGNN model on the ClimART dataset is as eeasy as:

```
python3 src/main.py -climart --train_single
```

All the configurations reguarging how the models are trained are found inside the `config.yaml` file.


### Contributions

Contributions are highly appreciated. If you find anything we can improve, please
go to the Discussions tab and create a New discussion, where you describe your suggestion.
For changes to code, download the repo, create a new branch off of latest\_release, 
and give it a new branch name. After making changes to code and/or adding new 
functionalities, push your code to the repo and create your pull request to latest\_release.

