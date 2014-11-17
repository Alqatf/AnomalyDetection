# -*- coding: utf-8 -*-
"""
http://www.astroml.org/sklearn_tutorial/dimensionality_reduction.html
"""

from sklearn.decomposition import RandomizedPCA

def reduction(df,n_components=2, random_state=0):
    rpca = RandomizedPCA(n_components=n_components, random_state=random_state)
    proj = rpca.fit_transform(df)
    return proj

