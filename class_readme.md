# CCS 578 Neural Collaborative Filtering

This is a tensorflow implementation for the neural network portion of the paper:

Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and Tat-Seng Chua (2017). [Neural Collaborative Filtering.](http://dl.acm.org/citation.cfm?id=3052569) In Proceedings of WWW '17, Perth, Australia, April 03-07, 2017.

Author: Dr. Xiangnan He (http://www.comp.nus.edu.sg/~xiangnan/)

## Environment Settings
- Tensorflow version:  '1.4'
- Python verions: '3.5'

## Installing the environment
Assuming you are using a debian distro, getting started is easy.  Just run ./install.sh to install anaconda3, create an environment with the project requirements, and activate
that environment.  You are then ready to run the code.  If you are not on a debian distro, you will have to install the requirements under 'Enviornment Settings' manually.

Train and evaluate the model on movielens data:
```
python NeuMF_tf.py --dataset ml-1m
```

Train and evaluate the model on pinterest data
```
python NeuMF_tf.py --dataset pinterest-20
```
