import sys
import pathlib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

import joblib
from eval_model import eval_model


def model_trainer(test_arr, train_arr, alpha, max_iter, l1_ratio, criterion, splitter, max_features,
                  n_estimators, learning_rate, loss, loss_g, subsample, criterion_g,
                  base_estimator, bagging_n_estimators, max_samples):
    X_train, y_train, X_test, y_test = (train_arr[:, :-1], train_arr[:, -1],
                                        test_arr[:, :-1], test_arr[:, -1])
    models = {
        'linearRegression': LinearRegression(),
        'ridge': Ridge(),
        'lasso': Lasso(),
        'elasticNet': ElasticNet(),
        'decisionTreeRegressor': DecisionTreeRegressor(),
        'randomForestRegressor': RandomForestRegressor(),
        'baggingRegressor': BaggingRegressor(),
        'adaBoostRegressor': AdaBoostRegressor(),
        'gradientBoostingRegressor': GradientBoostingRegressor()

    }

    param = {
        'linearRegression': {},
        'ridge': {
            'alpha': alpha,
            'max_iter': max_iter
        },
        'lasso': {
            'alpha': alpha,
            'max_iter': max_iter
        },
        'elasticNet': {
            'alpha': alpha,
            'max_iter': max_iter,
            'l1_ratio': l1_ratio
        },
        'decisionTreeRegressor': {
            'criterion': criterion,
            'splitter': splitter,
            'max_features': max_features
        },
        'randomForestRegressor': {
            'criterion': criterion,
            'max_features': max_features,
            'n_estimators': n_estimators
        },
        'baggingRegressor': {
            "base_estimator": base_estimator,
            "n_estimators": bagging_n_estimators,
            "max_samples": max_samples,
            "bootstrap": True,
            "max_features": max_features,
            "bootstrap_features": True,
            "random_state": 42
        },
        'adaBoostRegressor': {
            'learning_rate': learning_rate,
            'loss': loss,
            'n_estimators': n_estimators
        },
        'gradientBoostingRegressor': {
            'loss': loss_g,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'subsample': subsample,
            'max_features': max_features,
            'criterion': criterion_g
        }

    }

    model_report = eval_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models,
                                    params=param)

    model_df = pd.DataFrame.from_dict(model_report, orient='index')
    print(model_df)
    #model_df.to_csv('model_report.csv', index=False)

    df_new = model_df.sort_values(by=['adj_r2','r2_test'], ascending=False).reset_index()
    best_model_name = df_new['index'][0]

    print(f"Best model found, Model Name:{best_model_name}, R2 Score:{model_report[best_model_name]['r2_test']}, and Adjusted R2 Score:{model_report[best_model_name]['adj_r2']}")
    print("\n=========================\n")

def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parents[2]
    params_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))["train_model"]

    input_file = sys.argv[1]
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + '/models'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    train_dict = np.load(data_path + 'train_arr.npz')
    train_arr = train_dict['arr_0']
    test_dict = np.load(data_path + 'test_arr.npz')
    test_arr = test_dict['arr_0']

    model_trainer(train_arr=train_arr, test_arr=test_arr, alpha=params['alpha'], max_iter=params['max_iter'],
                  l1_ratio=params['l1_ratio'],
                  criterion=params['criterion'], splitter=params['splitter'], max_features=params['max_features'],
                  n_estimators=params['n_estimators'], learning_rate=params['learning_rate'], loss=params['loss'],
                  loss_g=params['loss_g'],
                  subsample=params['subsample'], criterion_g=params['criterion_g'],
                  base_estimator=params['base_estimator'], bagging_n_estimators=params['bagging_n_estimators'],
                  max_samples=params['max_samples']
                  )


if __name__ == "__main__":
    main()
