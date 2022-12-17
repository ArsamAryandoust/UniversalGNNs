# Universal graph neural networks for multi-task transfer learning 

We seek a universal graph representation of data that is able to capture arbitrary
events described by arbitary data types. Our goal is to develop a universal deep 
learning model that is then able to solve multiple tasks from multiple domains 
using a single model instance by sharing model parameters through a backbone graph 
neural network.

We want to test whether using a single model instance for solving multiple tasks 
can save computation and increase prediction accuracy compared to using different 
models on each task. Our intuiton is that we can leverage already performed computation
for solving some tasks for better solving another set of both related and unrelated 
tasks. If this is found to be true, we indicate that such a universal can achieve
a general task-independent level of artificial intelligence. 


We test this here on three prediction tasks that are important for enhancing the
global energy transition, and are unrelated in terms of the data they involve: 
first, the prediction of travel times between different city zones that are used 
to infer car parking density maps for the urban planning of electric mobility; 
second, the prediction of cost-effective materials for new storage technologies 
and solar fuels; third, the prediction of atmospheric radiative transfer that is 
crucial for weather and climate models.


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
