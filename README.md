# Enhancing AutoML for tackling climate change with universal deep learning models

Machine learning (ML) is increasingly found to have a significant potential for 
enhancing technologies that we urgently need for tackling climate change by solving 
a wide range of often complex prediction problems. Finding solutions requires domain 
experts without ML expertise from various fields like energy, mobility, policy 
and ecology to collaborate with ML experts for solving their individual prediction 
tasks, which is often difficult to realize and inefficient at scale. Automated 
ML (AutoML) reduces the need for human expertise for using ML by utilizing hyper 
parameter optimization, meta learning and neural architecture search. A recent 
evaluation of using AutoML for solving typical climate change related tasks, however, 
shows that existing tools fail to outperform human designed models because they 
are mostly designed with an emphasis on more mainstream tasks from the fields of 
computer vision and natural language processing, and are less suited for common 
climate change related themes like physics constrained ML or spatio-temporal data. 
A fundamental research question that therefore arises is: How do we enhance the 
current state of AutoML technology for tackling climate change related prediction 
problems?

We identify three important properties of climate change related prediction problems 
that need to be addressed in order to provide an answer to this question. First, 
prediction problems that tackle climate change in different domains can have complex 
but important relationships which can be leveraged to save computation and improve 
prediction accuracy, something that is dealt with by the field of transfer learning.
Second, solving multiple prediction tasks using the same ML model is difficult if 
different types and constellations of data are involved among these tasks; the 
field of multi-task learning deals with this type of problems by sharing the same 
prediction model parameters, which are often referred to as backbone neural networks, 
for improving the performance on multiple tasks simultaneously, as opposed to transfer 
learning where only performance in a target domain with sparse data availability 
is improved. Third, real-world events that are related to climate change are often 
captured by spatio-temporal data, influencing each other if they occur in close 
proximity in space and time and having variant distributions in both space and 
time, and other (hidden) conditions associated with these. These two properties 
of real world events break the traditional and theoretical assumption of data being 
identically and independently distributed (iid), which existing ML methods often 
rely on for guaranteed convergence. Because of these three properties, we hypothesize 
that universal deep learning (DL) models that are able to learn and infer from non-iid 
data in multiple domains and tasks can enhance the current state of AutoML technology 
for tackling climate change.

Recent approaches that have come closest to providing such universal prediction 
models are backbone neural networks that are able to solve multiple prediction 
tasks like image captioning, object detection and phrase grounding for vision-language 
processing; automated multi-task learners that compile user-provided backbone neural 
networks into DL models that are able to solve multiple vision tasks; and generalist 
agents that are able to play Atari games, chat, caption images, stack blocks with 
a real robot arm, and much more using a single large DL prediction model. However, 
none of these models has been designed or tested for solving a set of diverse 
prediction problems related to climate change, or proven to have performance 
guarantees for learning and inferring in non-iid data environments. 

Here, we want to explore universal methods that are able to solve important climate
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
docker run -it -v ~/UniversalDeepLearning:/UniversalDeepLearning -p 3333:1111 main_notebook
```

Compute using GPUs too:

```
docker run -it --gpus all -v ~/UniversalDeepLearning:/UniversalDeepLearning -p 3333:1111 main_notebook
```


