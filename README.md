# Explanatory-Paradigms
Code used in the paper [Explanatory Paradigms in Neural Networks](https://arxiv.org/abs/2202.11838), published in the Signal Processing Magazine

**Abstract** 

In this article, we present a leap-forward expansion to the study of explainability in neural networks by considering explanations as answers to abstract reasoning-based questions. With *P* as the prediction from a neural network, these questions are **Why P?**, **What if not P?**, and **Why P, rather than Q?** for a given contrast prediction *Q*. The answers to these questions are observed correlations, observed counterfactuals, and observed contrastive explanations respectively. Together, these explanations constitute the abductive reasoning scheme. We term the three explanatory schemes as observed explanatory paradigms. The term observed refers to the specific case of *post-hoc* explainability, when an explanatory technique explains the decision *P* after a trained neural network has made the decision *P*. The primary advantage of viewing explanations through the lens of abductive reasoning-based questions is that explanations can be used as reasons while making decisions. The post-hoc field of explainability, that previously only justified decisions, becomes active by being involved in the decision making process and providing limited, but relevant and contextual interventions. The contributions of this article are: (i) realizing explanations as reasoning paradigms, (ii) providing a probabilistic definition of observed explanations and their completeness, (iii) creating a taxonomy for evaluation of explanations, and (iv) positioning gradient-based complete explanainability's replicability and reproducibility across multiple applications and data modalities.

![Contrastive Explanations](Explanations.png) 

**Paper** : [Explanatory Paradigms in Neural Networks](https://arxiv.org/abs/2202.11838)

**Usage** : 

1. Version 1 (March 1 2022): The function methods.py consists of classes for [GradCAM](https://arxiv.org/abs/1610.02391) (observed correlations), [CounterfactualCAM](https://arxiv.org/abs/1610.02391) (observed counterfactuals), and [ContrastCAM](https://arxiv.org/abs/2008.00178) (observed contrast). All these methods are called from the main.py function for any of ResNet, VGG16, Densenet, AlexNet, and SqueezeNet architectures.

**Code Acknowledgements** :  The code is built on Grad-CAM. We use the implementation of **https://github.com/1Konny/gradcam_plus_plus-pytorch** as our base code. 

**Citation** : Please cite the following two papers if this code is used: 

1. [Contrastive Explanations in Neural Networks](https://arxiv.org/abs/2008.00178) : Prabhushankar, M., Kwon, G., Temel, D., & AlRegib, G. (2020, October). Contrastive explanations in neural networks. In 2020 IEEE International Conference on Image Processing (ICIP) (pp. 3289-3293). IEEE.
2. [Explanatory Paradigms in Neural Networks](https://arxiv.org/abs/2202.11838) : AlRegib, G., & Prabhushankar, M. (2022). Explanatory Paradigms in Neural Networks. arXiv preprint arXiv:2202.11838.
