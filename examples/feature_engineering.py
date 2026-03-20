"""Feature engineering with tunable hyperparameters."""

from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from lazyopt import hp

poly_degree = hp("poly_degree", "int", 1, values=[1, 2, 3])
use_interactions = hp("use_interactions", "int", 0, values=[0, 1])


def transform(X):
    POLY_DEGREE = int(poly_degree)
    USE_INTERACTIONS = int(use_interactions)

    if POLY_DEGREE > 1:
        interaction_only = not bool(USE_INTERACTIONS)
        poly = PolynomialFeatures(
            degree=POLY_DEGREE, interaction_only=interaction_only, include_bias=False
        )
        X = poly.fit_transform(X)

    X = StandardScaler().fit_transform(X)
    return X
