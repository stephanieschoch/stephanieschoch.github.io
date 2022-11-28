---
layout: post
title:  CS-Shapley &#58; Class-wise Shapley Values for Data Valuation in Classification
date: 2022-11-28 12:00:00
description: A blog post about the NeurIPS 2022 paper "CS-Shapley &#58; Class-wise Shapley Values for Data Valuation in Classification"
tags: data-valuation
categories: blog-posts
---
      
# Data Contribution Estimation:
In machine learning settings, there are notable benefits of understanding how individual training instances impact a learning model. For example, through identifying and filtering points that harm the model (e.g. noisy or mislabeled instances), the performance on a subsequent model retraining may increase. We could additionally seek to augment the data by identifying new data instances that are similar to training instances that were highly beneficial to the model. In this setting, we can refer to how an instance impacts a performance metric of choice as the “contribution” of the data point.

The prudent question to ask then becomes one of how to measure this contribution. While we could simply measure the model performance when trained with vs. without the training instance (i.e. Leave-One-Out from Cook, 1977), this method has certain drawbacks as it does not satisfy several properties desirable for measuring data contributions and does not always perform as expected in practice. Ghorbani et al. (2019) exemplified this with a concrete example: if we are measuring contribution to a KNN classifier and have two copies of each data point, removal of one point would not change the classifier performance, and each data point would receive a contribution score of $0$. 

## Shapley Values: 
Shapley values (Shapley, 1953) have been proposed for use in this context and have proven to be effective for measuring data contributions, and the associated applications. Shapley values, from cooperative game theory, satisfy desirable fairness guarantees due to their underlying axiomatic basis. For a value function $v(·)$, the Shapley value $\phi i(T, A, v)$, for any data point $i$ is defined as:

$$
\phi_i(T, \mathcal{A}, v)= \sum\limits_{S \subseteq T\textbackslash\{i\}} \frac{v(S\cup\{i\})-v(S)}{\binom{n-1}{\mid S \mid}}
$$

In simple terms, the Shapley value of a data point measures its average marginal contribution to every possible data subset.

Much of the work in applying Shapley values to data contribution measurement, or data valuation, has sought to develop approximation techniques to mitigate the computational cost of true Shapley computation. Specifically, true Shapley value computation is exponential with respect to the number of data points, and as such, entails an exponential number of model retrainings. One such approximation method is the Truncated Monte Carlo method proposed by Ghorbani et al. (2019), which we adopt in this work. Additional approximation methods can be found listed in the paper.

# CS-Shapley: 
## Intuition:
What the existing methods have in common, is how the value function underlying Shapley computation is defined. More specifically, the value function is defined over the entire development set (in practice, development accuracy). In this work, we challenge the implicit assumption that full development set metrics are ideal for Shapley computation on classification datasets. Our intuition was that defining the value function in this manner may have limited ability to differentiate helpful or harmful training instances. We provide an example in Figure 1 below.

![](/assets/img/cs-shapley-fig-1.png)

While we provide more details in the paper, in short, this example shows two training points from the real world CIFAR10 datasets that belong to the same class, cause the same overall development accuracy change, yet data point I increases in class accuracy while data point j decreases in-class accuracy. Intuitively, data points that harm their own classes may be mislabeled or otherwise noisy.	

## CS-Shapley Definition:

To address this, we define a value function that uses in-class accuracy as a measure of contribution and out-of-class accuracy as a weighting, or discounting, factor. 

Formally, we define the value function $$v(.)$$ as 

$$v_{y_i}(S_{y_i}|S_{-y_i}) = a_S(D_{y_i})\cdot e^{a_S(D_{-y_i})}$$

where $$a_S(D_{y_i})$$ indicates in-class accuracy and $$a_S(D_{-y_i})$$ indicates out-of-class accuracy. With this, we can then define the **CS-Shapley value** of a data point $$i$$ as

$$\phi_i|S_{-y_i} = \sum_{S_{y_i} \subseteq T_{y_i} \setminus \{i\}} \frac{v_{y_i}(S_{y_i}\cup\{i\}| S_{-y_i})-v_{y_i}(S_{y_i}| S_{-y_i})}{\binom{n-1}{\mid S_{y_i} \mid}}$$

While we demonstrate several desirable properties of this function in the paper, we can illustrate this function in the following contour plot:

![](/assets/img/fig-cd-contourplot.png)

The effect of the out-of-class accuracy is controlled by the value of the in-class accuracy. In other words, when the in-class accuracy is low, the out-of-class accuracy can essentially be ignored. Conversely , when the in-class accuracy is high, the out-of-class accuracy can have a substantial effect on the valuation of an in-class data point.

In [the paper](https://arxiv.org/pdf/2211.06800.pdf), we demonstrate the efficacy of this value function applied to Shapley-based data valuation using three tasks: high-value data removal, noisy data detection, and transferability of data values. Please see our paper for more details!

# References: 

R Dennis Cook. Detection of influential observation in linear regression. *Technometrics*, 19(1):
15–18, 1977.

Amirata Ghorbani and James Zou. Data shapley: Equitable valuation of data for machine
learning. In *International Conference on Machine Learning*, pages 2242–2251. PMLR, 2019.

Lloyd S Shapley. A value for n-person games, contributions to the theory of games, 2, 307–317, 1953
