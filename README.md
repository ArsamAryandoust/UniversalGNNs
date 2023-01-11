# Universal graph neural networks for multi-task learning

Deep learning (DL) models for multi-modal multi-task learning are increasingly found to have the ability to assist us in solving sophisticated real world problems. The most recent and powerful representatives are large language based models like Gato developed at Deep Mind, which is able to play Atari, caption images, chat, stack blocks with a real robot arm and much more using a single generalist DL agent, and ChatGPT developed at Open AI, which is able to write and debug code in multiple languages, derive mathematical theorems and much more, using again a single DL model that learns from interactions with humans. A natural question that arises from these developments is whether multi-modal multi-task DL models, which we hereafter simply call *universal* DL models, are also able to assist us in solving more important and urgent problems such as for example the many challenges involved in enhancing the global energy transition and mitigating climate change. 


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
