### Introduction 
---
This section will contain my notes of the course [XCS236](https://online.stanford.edu/courses/xcs236-deep-generative-models) - Deep Generative Models. I would be including some information I got from reading the [text book from OReilly](https://learning.oreilly.com/library/view/generative-deep-learning/9781098134174) as well. That book is on the same subject. So here I go ...

### What is Generative Modelling 
---
 
Generative modeling is a branch of machine learning that involves training a model to produce new data that is similar to a given dataset.

### Example Use Cases 
---

##### Image Generation & Control
Generative models can create realistic images based on rough sketches or other control signals.
These models help users who may not be skilled artists to generate high-quality visual content.

##### Audio Applications
Similar techniques apply to generative models for audio, where input signals can shape the generated sound.
Examples include voice synthesis and music composition.

##### Enhancements & Applications
Generative models are used in tasks like style transfer, content creation, and data augmentation.
They enable creative control, allowing for transformation of inputs into complex, high-quality outputs.

### Generative Versus Discriminative Modeling
---
Discriminative modeling estimates *p(y|x)* i.e. discriminative modeling aims to model the probability of a label given some observation. Generative modeling estimates *p(x)* i.e.generative modeling aims to model the probability of observing an observation. Sampling from this distribution allows us to generate new observations.

### Modelling High Level View - What do we do in the modelling process?
---
- We have a dataset of observations 
- We assume that the observations have been generated according to some unknown distribution, *p<sub>data</sub>*
- We want to build a generative model *p<sub>theta</sub>*
 that mimics *p<sub>data</sub>*
- If we achieve this goal, we can sample from 
 to generate observations that appear to have been drawn from *p<sub>data</sub>*


### What all we can do once we have learn a generative model 
---
1. Generation: It should be possible to easily sample a new observation from *p<sub>theta</sub>* and the generated sample should look as if it was generated from *p<sub>data</sub>* (sampling)
2. Density estimation: *p<sub>theta</sub>(x)* should be high if x looks like being generated from *p<sub>data</sub>*, and low otherwise (anomaly detection)
3. Unsupervised representation learning: We should be able to learn the latent features of the data. In case of images it means what the images have in common. For e.g. in case of images of animals, features can be ears, tail, etc. (features)
