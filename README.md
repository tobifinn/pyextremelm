#pyextremelm
pyextremelm is a **Python** module which implements the
**Extreme Learning Machine** [1], in the style of **scikit-learn** [2]. </br>
This repository is in the alpha (V. 0.1) state. It will grow with the time.

**Project description:**</br>
This package implements the so called Extreme Learning Machine (ELM) for Python.
Firstly the ELM was a Single Hidden-layer Feedforward Neural Network (SLFN),
but nowadays there are methods to implement a multilayer network based ELM.</br>
The main idea is that the weights between the input and the hidden neurons
don't need to be trained to fit any function.</br>

In this module are two different possibilities to fit an ELM.</br>
One possibility is
to use the **pre-configured networks**, were only the number of hidden neurons have
to be set, but also other parameters could be set.</br>
The other possibility is to **configure your own ELM network with the
builder**. There you have more possibilities to create different types of networks
(e.g. deep ELM networks) and you can also combine different layers. The main
idea of the builder is that the **differences between different layers are
the different training approaches** (e.g. the difference between a
regression output layer and an hidden layer are only the linear regression and
the random weights, but they have both neurons, which needs to be trained).</br>

Due to the fact, that the networks have **almost the same syntax as the
scikit-learn module** you can insert such a network into the pipline module of
scikit-learn.
Some basic examples (at the moment only a simple sinus regression and the use
of the auto-encoder for images) could be found in the examples folder.

This project is rapidly developing, so some of the code
hasn't any docstring yet and also the documentation isn't complete.


**Currently implemented:**
<li>Pre-configured networks (only regression currently).
<li>Network builder to generate own networks.
<li>Regression learning based on the approach without and with constrainment [1,3,4].
<li>Different activation functions.
<li>Random algorithm to get the best set of random weights.
<li>ELM-autoencoder [7].
<li>Orthogonal random weights.

**Planned:**
<li>Classification learning with and without constrainment [1,3,4].
<li>Supervised learning with any supervised scikit-learn function.
<li>Unsupervised learning based on [5,6].
<li>Support for unsupervised scikit-learn algorithms.
<li>Constrainment based on the L1-norm (at the moment the lasso of scikit-learn
is used for sparse regression) [8]
<li>Local receptive fields based extreme learning machine [9]
<li>Plotting function to visualize the Extreme Learning Machine with training results.



Requirements
------------
Written using Python 3.5.<br>
It requires:
<li>Numpy
<li>Scipy
<li>Scikit-learn
<li>Matplotlib (only for future purpose)



References
----------
```
[1] Guang-Bin Huang, Qin-Yu Zhu and Chee-Kheong Siew,
        "Extreme learning machine: a new learning scheme of feedforward neural networks",
        Neural Networks, 2004. Proceedings. 2004 IEEE International Joint Conference on,
        2004, pp. 985-990, vol.2.

[2] Pedregosa et al.,
        "Scikit-learn: Machine Learning in Python"
        Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.

[3] G.-B. Huang, Q.-Y. Zhu and C.-K. Siew,
        "Extreme Learning Machine: Theory and Applications",
        Neurocomputing, vol. 70, pp. 489-501, 2006.

[4] Huang, Guang-Bin, et al.
        "Extreme learning machine for regression and multiclass classification."
        Systems, Man, and Cybernetics, Part B: Cybernetics,
        IEEE Transactions on 42.2 (2012), pp. 513-529.

[5] Huang, Gao, et al.
        "Semi-supervised and unsupervised extreme learning machines."
        Cybernetics, IEEE Transactions on, 2014, 44. Jg., Nr. 12, pp. 2405-2417.

[6] Lekamalage, Chamara Kasun Liyanaarachchi, et al.
        "Extreme learning machine for clustering."
        Proceedings of ELM-2014, vol. 1.,
        Springer International Publishing, 2015. pp. 435-444.

[7] E. Cambria et al.,
        "Extreme Learning Machines [Trends & Controversies],"
        in IEEE Intelligent Systems, vol. 28, no. 6, pp. 30-59, Nov.-Dec. 2013.

[8] Tang, Jiexiong, Chenwei Deng, and Guang-Bin Huang.,
        "Extreme learning machine for multilayer perceptron." (2015).

[9] Huang, Guang-Bin, et al.
        "Local receptive fields based extreme learning machine."
        Computational Intelligence Magazine, IEEE 10.2 (2015): 18-29.

```

