import sys, pandas
from _utilities import *

# Classifiers
# All the available classifiers, classifier is the function that sets which is used
@print_state("[T] Training SVC Model...")
def classifier_svc(data, labels):
    from sklearn.svm import SVC
    m = svm.SVC(gamma="scale")
    return m.fit(data, labels)

@print_state("[T] Training NaiveBayes Model...")
def classifier_nb(data, labels):
    from sklearn.naive_bayes import GaussianNB
    m = GaussianNB()
    return m.fit(data, labels)

@print_state("[T] Training LogisticRegression Model...")
def classifier_lr(data, labels):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    return m.fit(data, labels)

@print_state("[T] Training Neural Network...")
def classifier_nn(data, labels):
    from sklearn.neural_network import MLPClassifier
    m = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    return m.fit(data, labels)

classifier = timer(classifier_nn)
