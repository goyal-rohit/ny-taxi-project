from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def eval_model(X_train, y_train, X_test, y_test, models, params):
    report = {}

    for i in range(len(list(models))):
        model = list(models.values())[i]
        param = params[list(models.keys())[i]]

        print('training....', model)
        gs = GridSearchCV(model, param, cv=5,n_jobs=-1)
        gs.fit(X_train, y_train)

        model.set_params(**gs.best_params_)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_score_r2 = r2_score(y_train, y_train_pred)
        test_score_r2 = r2_score(y_test, y_test_pred)

        n, p = X_test.shape

        adj_r2 = 1 - (1 - test_score_r2) * (n - 1) / (n - p - 1)
        #print(train_score_r2, test_score_r2, adj_r2)

        report[list(models.keys())[i]] = {'r2_train': train_score_r2, 'r2_test': test_score_r2, 'adj_r2': adj_r2}

    return report


