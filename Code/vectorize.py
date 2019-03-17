import sys, pandas
from _utilities import *
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer

_print_results = lambda o,*a,**k: "got {1} features for {0} mails.".format(*o[1].shape)
# Vectorizer Options
# All the types of Vectorizer, the one that is set as extract_features is the one used
@print_state("[#] Extracting Features using CountVectorizer...", text_after_f=_print_results)
def vectorize_count(mails):
    vec = CountVectorizer()
    data = vec.fit_transform(mails)
    return (vec, data)

@print_state("[#] Extracting Features using HashingVectorizer...", text_after_f=_print_results)
def vectorize_hash(mails):
    vec = HashingVectorizer(n_features=2**10)
    data = vec.fit_transform(mails)
    return (vec, data)

@print_state("[#] Extracting Features using TfIdfVectorizer...", text_after_f=_print_results)
def vectorize_tfidf(mails):
    vec = TfidfVectorizer()
    data = vec.fit_transform(mails)
    return (vec, data)

extract_features = timer(vectorize_hash) # Set the preferred Vectorizer based feature extraction
