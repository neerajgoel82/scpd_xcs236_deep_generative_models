### Introduction 
This section will contain my notes of the course [XCS236](https://online.stanford.edu/courses/xcs236-deep-generative-models) - Deep Generative Models. I would be including some information I got from reading the [text book from OReilly](https://learning.oreilly.com/library/view/generative-deep-learning/9781098134174) as well. That book is on the same subject. So here I go ...

### What is Generative Modelling
Generative modeling is a branch of machine learning that involves training a model to produce new data that is similar to a given dataset.

### High Level View - What do we do in the modelling process?
- We have a dataset of observations 
- We assume that the observations have been generated according to some unknown distribution, *p<sub>data</sub>*
- We want to build a generative model *p<sub>theta</sub>*
 that mimics *p<sub>data</sub>*
- If we achieve this goal, we can sample from 
 to generate observations that appear to have been drawn from *p<sub>data</sub>*


### Generative Versus Discriminative Modeling
Discriminative modeling estimates p(y|x) i.e. discriminative modeling aims to model the probability of a label given some observation. Generative modeling estimates p(x) i.e.generative modeling aims to model the probability of observing an observation. Sampling from this distribution allows us to generate new observations.


