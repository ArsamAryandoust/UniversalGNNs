# Universal graph neural networks for multi-task learning

Given arbitrary prediction tasks and datasets, we want to train a single DL model that is able to solve them all. We propose a model that we call **UniversalGNN** for this, which is composed of four different components: an auto-encoder, a graph builder, a backbone GNN and a final regressor or classifier. The backbone GNN is the only component that is shared across all tasks, while the design of the other components depends on the individual tasks such that we create one for each dataset.

## Download
Download this repository and the `EnergyTransitionTasks` one to your home directory:

```
cd 
git clone https://github.com/ArsamAryandoust/UniversalGNNs.git
git clone https://github.com/ArsamAryandoust/EnergyTransitionTasks.git
cd UniversalGNNs
```

## Docker

The easiest way to build and run the Docker container is with the `build_and_run.sh` script inside the `UniversalGNNs` folder.

## Getting started

All the models (included the baselines) can be trained from the `main.py` file inside of the `src` folder. The easiest way to do so is to start the Docker container built as above and inside of it run:

```
python3 src/main.py --help
```

This will show the available command line arguments that control the models that will be trained. For example, training the UniversalGNN model on the ClimART dataset is as eeasy as:

```
python3 src/main.py -climart --train_single
```

All the configurations reguarging how the models are trained are found inside the `config.yaml` file.
