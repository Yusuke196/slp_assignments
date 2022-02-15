# Part-of-speech tagging

This directory is for a project to perform supervised part-of-speech tagging.


## Approaches

I created a first-order HMM model, in which POS tags are approximately predicted by the following equation:

$$ \hat{t}_{1:n} = argmax_{t_i \cdots t_n} \prod_{i = 1}^n P(w_i|t_i)P(t_i|t_{i - 1}) $$

where $P(w_i|t_i)$ is called emission probability and $P(t_i|t_{i - 1})$ transition probability. In order to calculate the emission probability when $w_i$ is unseen in the training data, I performed the following smoothing:

$$ P(w_i|t_i) = \lambda P(w_i|t_i) + (1 - \lambda) \frac{1}{N_{t_i}} $$

where $N_{t_i}$ is the vocabulary size calculated as the unique number of tokens emitted from $t_i$ plus $1$, which represents `<UNK>`.

To explore the appropriate value for $\lambda$, I conducted hyperparameter tuning using `tuning.py`. As a result, the highest accuracy was obtained when $\lambda = 0.99999$.

## Results

The highest accuracy achieved was $86.87%$.


---


## Install Packages

```
pip install -r requirements.txt
```

The project is tested by Python 3.9.9.


## Download Data

```
cd data
wget https://raw.githubusercontent.com/neubig/nlptutorial/master/data/wiki-en-train.norm_pos
wget https://raw.githubusercontent.com/neubig/nlptutorial/master/data/wiki-en-test.norm_pos
```


## Train and Predict

```
python pos_tagging.py
```


## Evaluate

```
eval/gradepos.pl eval/pred.txt eval/wiki-en-test.pos
```

The program `gradepos.pl` was obtained [here](https://github.com/neubig/nlptutorial/tree/master/script).
