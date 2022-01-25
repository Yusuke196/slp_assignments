## 1. Traditional Language model



## 2. Class Prediction

I created 2 models; first one is named Rule Base, and second one is Naive Bayes. Their accuracies on the test sets are as follows.

|          | Rule Base | Naive Bayes |
| -------- | --------- | ----------- |
| accuracy | 0.886     | 0.729       |

Rule Base model uses regular expression to find 2 types of expressions that appear in explanations on persons frequently. One of them is the person's name, and the other is expressions on the date, month or year of birth or death.

On the other hand, Naive Bayes model assumes that each word in the sentences appear independently of other words, and the order in which each word appears can be ignored. In order to tackle the zero probability problem of unseen events, the add-one (Laplace) smoothing is performed.

One finding on this Naive Bayes model is that the highest accuracy was gained when no rare words are deleted from the voabulary set, though it is computationally heavy; the accuracy dropped to 0.495 when words that appeared less than or equal to 3 times are deleted.

For the program codes, please refer to class_pred.py.
