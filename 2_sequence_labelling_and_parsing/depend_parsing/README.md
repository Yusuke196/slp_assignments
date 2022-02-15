# Report

This directory is for a project to perform supervised dependency parsing.


## Approaches

I created a model of Shift-Reduce method, adopting Support Vector Machine (SVM) as the classifier. The learning phase to build the classifier consists following steps.

1. Transform the train data from token-base to step-base, where we have a matrix each row of which contains last two tokens of the stack, their POS, first token of the buffer, its POS, and the action. The action is our objective and the rest is used as features to predict it.
1. Initialize a SVM classifier and make it learn to predict an appropriate action. The features mentioned above are used here after one hot encoding.

Then parsing on the test data is done as follows.

1. Apply Shift-Reduce method on the test data using the trained classifier. That is, when prediction is done on a configuration (a set of features explained above), set another configuration based on the prediction.
1. Repeat the above operation until the buffer is vacant and the contents of the stack are only 2 ROOT tokens.


## Results

I made predictions using SVM classifiers. Their kernels and the resulting accuracies are:

|          | Radial Basis Function | Polynomial | Linear |
| -------- | --------------------- | ---------- | ------ |
| accuracy | 61.61                 | 71.16      | 67.99  |


---


## Install Packages

```
pip install -r requirements.txt
```

The project is tested by Python 3.9.9.


## Download Data

```
cd data
wget https://raw.githubusercontent.com/neubig/nlptutorial/master/data/mstparser-en-train.dep
wget https://raw.githubusercontent.com/neubig/nlptutorial/master/data/mstparser-en-test.dep
```


## Train

```
python train.py --kernel=rbf/poly/linear
```


## Predict

```
python predict.py --kernel=rbf/poly/linear
```


## Evaluate

```
python2 eval/grade-dep.py data/mstparser-en-test.dep eval/<prediction file>
```

The program `grade-dep.py` was obtained [here](https://github.com/neubig/nlptutorial/tree/master/script).
