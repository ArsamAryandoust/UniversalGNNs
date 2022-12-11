# Enhancing AutoML for tackling climate change with universal deep learning models

We explore universal deep learning models that are able to solve important climate
change related prediction tasks. We demonstrate these here on three prediction tasks:
first, the prediction of travel times between different city zones that are used 
to infer car parking density maps for the urban planning of electric mobility; 
second, the prediction of cost-effective materials for new storage technologies 
and solar fuels; third, the prediction of atmospheric radiative transfer that is 
crucial for weather and climate models.


## Download
Download this repository to your home directory:

```
cd 
git clone https://github.com/ArsamAryandoust/UniversalDeepLearning
cd UniversalDeepLearning
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
docker run -it -v ~/UniversalDeepLearning:/UniversalDeepLearning -p 3333:1111 main_notebook
```

Compute using GPUs too:

```
docker run -it --gpus all -v ~/UniversalDeepLearning:/UniversalDeepLearning -p 3333:1111 main_notebook
```


