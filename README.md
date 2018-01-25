# ladder_network
This project proposes an implementation of the ladder network described by Rasmus et al. (https://arxiv.org/abs/1507.02672).

The ladder network combines supervised and unsupervised learning: the classifier is trained either with labeled and unlabeled examples.
It is an interesting option when a large amount of data is available but a small part of it is labeled.

The design of the program is inspired by the implementation of rinuboney (https://github.com/rinuboney/ladder). Modifications were done to add convolution and max pooling layers. 
The architecture of the algorithm is designed in a flexible way, in order to make the modification easy (add layers...)

The projects proposes two applications of the ladder network: the MNIST hand-written digits and the Statoil challenge (https://www.kaggle.com/c/statoil-iceberg-classifier-challenge).
