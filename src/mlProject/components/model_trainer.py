import pandas as pd
import os
import joblib

from mlProject import logger
from mlProject.entity.config_entity import ModelTrainerConfig

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV  


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)

        train_y = train_data[self.config.target_column]
        test_y = test_data[self.config.target_column]

        logger.info("Data loaded and split into train/test sets")

        rf = RandomForestRegressor(
            random_state=42,
            n_jobs=-1
        )

        param_dist = {
            "n_estimators": self.config.n_estimators,
            "max_depth": self.config.max_depth,
            "min_samples_split": self.config.min_samples_split,
            "min_samples_leaf": self.config.min_samples_leaf,
            "max_features": self.config.max_features
        }

        rf_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=30,
            cv=5,
            scoring="neg_root_mean_squared_error",
            verbose=2,
            random_state=42,
            n_jobs=-1
        )

        logger.info("Starting RandomForest hyperparameter tuning")
        rf_search.fit(train_x, train_y)

        best_model = rf_search.best_estimator_

        logger.info(f"Best RandomForest params: {rf_search.best_params_}")
        logger.info(f"Best CV RMSE: {-rf_search.best_score_}")

        model_path = os.path.join(self.config.root_dir, self.config.model_name)
        joblib.dump(best_model, model_path)

        logger.info(f"Best model saved at: {model_path}")
