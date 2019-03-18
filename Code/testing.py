from sklearn.model_selection import KFold

from _utilities import *

def splitDataframe(df, splits=3):
    return KFold(n_splits=splits).get_n_splits(df)
