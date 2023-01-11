# Universal graph neural networks for multi-task learning

Deep learning (DL) models for multi-modal multi-task learning are increasingly found to have the ability to assist us in solving sophisticated real world problems. The most recent and powerful representatives are large language based models like Gato developed at Deep Mind, which is able to play Atari, caption images, chat, stack blocks with a real robot arm and much more using a single generalist DL agent, and ChatGPT developed at Open AI, which is able to write and debug code in multiple languages, derive mathematical theorems and much more, using again a single DL model that learns from interactions with humans. A natural question that arises from these developments is whether multi-modal multi-task DL models, which we hereafter simply call *universal* DL models, are also able to assist us in solving more important and urgent problems such as for example the many challenges involved in enhancing the global energy transition and mitigating climate change. 

Machine learning (ML) is generally found to have a significant potential for enhancing technologies that we urgently need for climate change mitigation and adaptation by solving a wide range of often complex prediction problems. The literature on tackling climate change with ML, however, is still very young and does not examine the role of universal DL models. A recent study that has come closest to this examines automated ML (AutoML) frameworks for tackling climate change related prediction problems, and finds that existing frameworks fail to outperform human designed models because they are mostly designed with an emphasis on more mainstream tasks from the fields of computer vision and natural language processing, and are less suited for common climate change related themes like physics constrained ML and spatio-temporal data. AutoML reduces the need for human expertise for using ML by utilizing hyper parameter optimization, meta learning and neural architecture search. Since tackling climate change with ML requires domain experts without ML expertise from various fields like energy, mobility, policy and ecology to collaborate with ML experts for solving their individual prediction tasks, AutoML frameworks may, similar to universal DL models, be suitable assistants to domain experts where collaborations are difficult to realize and inefficient at scale. A fundamental research question that therefore remains open is how universal DL models perform on solving typical climate change related tasks and whether they may be useful for enhancing AutoML technology. 

In this repository, we design a universal graph representation of data and a universal graph neural network (GNN) prediction model that we call UniversalGNN, which is able to solve multiple tasks with arbitrary data types. We investigate if solving several tasks with one instance of our UniversalGNN can improve performance compared with solving each task with a separate instance of our UniversalGNN. For this, we conduct two sets of experiments on three different prediction tasks. In a first set of experiments, we solve each prediction task with a separate instance of our UniversalGNN. This represents the current default method of choice and serves us as a baseline. In a second set of experiments, we train another instance of our UniversalGNN for solving all three prediction tasks simultaneously, and compare performance against our baselines. All models are evaluated against the performance of a random forest (RF) benchmark and are further compared with the performance of a simple multi-layer perceptron (MLP).


## Download
Download this repository to your home directory:

```
cd 
git clone https://github.com/ArsamAryandoust/UniversalGNNs
cd UniversalGNNs
```

## Docker

Build main Docker container:

```
docker build -t main Docker
```


## Jupyter notebooks inside Docker containers

Build Jupyter notebook container:

```
docker build -t main_notebook DockerNotebook
```

Compute using CPU only:

```
docker run -it -v ~/UniversalGNNs:/UniversalGNNs -p 3333:1111 main_notebook
```

Compute using GPUs too:

```
docker run -it --gpus all -v ~/UniversalGNNs:/UniversalGNNs -p 3333:1111 main_notebook
```

Open the link that shows in your terminal with a browser. Then, replace the port 
1111 with 3333 in your browser link to see notebooks inside the Docker container.
