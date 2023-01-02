# Universal graph neural networks for multi-task learning

We seek a universal graph representation of data that is able to capture arbitrary
events described by arbitary data types. Our goal is to develop a universal deep 
learning model that is then able to solve multiple tasks from multiple domains 
using a single model instance by sharing model parameters through a backbone graph 
neural network.

We want to test whether using a single model instance for solving multiple tasks 
can save computation and increase prediction accuracy compared to using different 
models on each task. We hypothesize that already performed computation for solving 
some tasks can be leveraged for better solving other sets of related and unrelated 
tasks. If this is found to be true, our results will indicate that multi-modal
multi-task deep learning models can achieve a general level of task-independent 
intelligence that needs to be further explored. 

In the following, we empirically test our hypothesis on three prediction tasks that 
are important for enhancing the global energy transition, and are independent from
each other in terms of the data they involve: first, we predict travel times between 
different city zones that are measured by Uber and are used to infer car parking 
patterns for the urban planning of electric mobility; second, we predict atmospheric 
radiative transfer that is modeled by the Canadian Earth System Model and is important
for reducing the sparsity of such physiscs informed models due to their high
computational complexity.


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
