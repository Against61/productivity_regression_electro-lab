from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV


def model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

    parameteres = {
        "fit_intercept": ['True'],
        'l1_ratio':[1.0],
        'alpha':[0.001, 0.0001, 0.01, 0.1, 0.00001]
    }

    grid = GridSearchCV(ElasticNet(copy_X=True), param_grid=parameteres, cv = 5, n_jobs=-1, scoring='r2', verbose=0)
    grid.fit(X_train, y_train)
    return grid
