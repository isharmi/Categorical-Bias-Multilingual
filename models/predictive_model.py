# models/predictive_model.py

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


class BiasPredictor:
    """
    Wrapper class for initializing and using multiple ML regression models.
    """

    def __init__(self, model_name: str, **params):
        """
        model_name: str
            Name of the model (e.g., 'rf', 'xgb', 'svr', 'mlp', etc.)
        params: dict
            Additional parameters passed to the model constructor.
        """

        self.model = self._create_model(model_name, params)

    def _create_model(self, name, params):
        name = name.lower()

        if name == 'linear':
            return LinearRegression(**params)
        if name == 'dt':
            return DecisionTreeRegressor(**params)
        if name == 'rf':
            return RandomForestRegressor(**params)
        if name == 'svr':
            return SVR(**params)
        if name == 'knn':
            return KNeighborsRegressor(**params)
        if name == 'mlp':
            return MLPRegressor(max_iter=200, **params)
        if name == 'ada':
            return AdaBoostRegressor(**params)
        if name == 'et':
            return ExtraTreesRegressor(**params)
        if name == 'gbr':
            return GradientBoostingRegressor(**params)
        if name == 'xgb':
            return XGBRegressor(
                objective='reg:squarederror',
                eval_metric='rmse',
                **params
            )
        if name == 'lgbm':
            return LGBMRegressor(**params)
        if name == 'cat':
            return CatBoostRegressor(
                verbose=False,
            )

        raise ValueError(f"Unknown model name: {name}")

    def fit(self, X, y):
        """Train the model."""
        self.model.fit(X, y)

    def predict(self, X):
        """Predict with the trained model."""
        return self.model.predict(X)
