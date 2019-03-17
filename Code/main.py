print("[§] Importing modules...")
import sys
from _utilities import *
from processing import load
from vectorize import extract_features
from classification import classifier

if __name__ == '__main__':
    mails = load(sys.argv[1])
    _, features = extract_features(mails["message"])
    classifier = classifier(features, mails["needs_reply"])

    # Do something with Classifier
