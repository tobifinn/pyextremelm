#pyextremelm
pyextremelm is a **Python** module which implements the
**Extreme Learning Machine** (ELM, [1]), in the style of **scikit-learn** [2]. </br>
This repository is in the alpha (V. 0.1) stadium. It will grow with the time.

**Currently implemented:**
<li>Regression learning based on the approach without any constrainment [1,3].
<li>Supervised learning with any scikit-learn function.
<li>Two different activation functions (sigmoid and tanh).
<li>Random algorithm to get the best set of random weights.

**Planned:**
<li>Any supervised learning with and without constrainment [1,3,4].
<li>Unsupervised learning based on [5,6].
<li>Support for unsupervised scikit-learn algorithms.
<li>ELM-autoencoder [7].
<li>More activation functions.


Requirements
------------
Written using Python 3.5.<br>
It requires:
<li>Numpy
<li>Scikit-learn



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

[3] G.-B. Huang, Q.-Y. Zhu and C.-K. Siew, "Extreme Learning Machine:
        Theory and Applications", Neurocomputing, vol. 70, pp. 489-501,
        2006.

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
```

