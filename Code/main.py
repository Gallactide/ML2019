print("[§] Importing modules...")
import sys
from _utilities import *
from processing import load
from vectorize import extract_features
from classification import classifier
from sklearn

# DEBUG
from sklearn.model_selection import KFold

from _utilities import *

def splitDataframe(df, splits=3):
    return KFold(n_splits=splits).get_n_splits(df)
#DEBUG

if __name__ == '__main__':
    mails = load(sys.argv[1])
    _, features = extract_features(mails["message"])
    classifier = classifier(features, mails["needs_reply"])

    # Do something with Classifier
