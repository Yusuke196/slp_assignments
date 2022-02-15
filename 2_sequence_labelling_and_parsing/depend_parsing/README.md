---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.6
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Report

This directory is for a project to perform supervised dependency parsing.


## Approaches

I created a model of Shift-Reduce method, adopting Support Vector Machine (SVM) as the classifier. The learning phase to build the classifier consists following steps.

1. Transform the train data from token-base to step-base, where we have a matrix each row of which contains last two tokens of the stack, their POS, first token of the buffer, its POS, and the action. The action is our objective and the rest is used as features to predict it.
1. Initialize a SVM classifier and make it learn to predict the appropriate action. The features are used here after one hot encoding is performed on them.

Then parsing on the test data is done as follows.

1. Apply Shift-Reduce method on the test data using the trained classifier. Prediction is done on a configuration, and another configuration is given based on the prediction, where a configuration means a set of features explained above.


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
