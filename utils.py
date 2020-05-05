import os
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.model_selection import cross_validate


PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def load_sensors_data(data_path='data/sensors.csv', replace_hyphen=False):
    # Importing the dataset
    df = pd.read_csv(data_path)

    if replace_hyphen is True:
        def replaceNA(value):
            if value == '-':
                return np.nan
            else:
                return value
        df = df.applymap(replaceNA)

    df.dropna(inplace=True)

    hum_df = df[['hum1', 'hum2', 'hum3', 'hum4']]
    temp_df = df[['temp1', 'temp2', 'temp3', 'temp4']]
    light_df = df[['light1', 'light2', 'light3', 'light4']]

    # print('hum features :', list(hum_df.columns))
    # print('temp features :', list(temp_df.columns))
    # print('light features :', list(light_df.columns))

    df_avg = df.loc[:, ['people', 'co2']]
    df_avg['hum_avg'] = hum_df.mean(axis=1)
    df_avg['temp_avg'] = temp_df.mean(axis=1)
    df_avg['light_avg'] = light_df.mean(axis=1)

    return df, df_avg


def cross_val(regressor, X, y, cv=10, verbose=1):
    scoring = {
        'explained_variance': 'explained_variance',
        'r2': 'r2',
        'abs_error': 'neg_mean_absolute_error',
        'squared_error': 'neg_mean_squared_error'
    }
    scores = cross_validate(regressor, X, y, cv=10,
                            scoring=scoring, return_train_score=True, verbose=verbose)

    print("Explained Variance :", scores['test_explained_variance'].mean())
    print("R2 :", scores['test_r2'].mean())
    print("MAE :", abs(scores['test_abs_error'].mean()))
    print("RMSE :", math.sqrt(abs(scores['test_squared_error'].mean())))

    return scores


def val_metrics(y_true, y_pred, prefix='Testing'):
    test_mse = mean_squared_error(y_true, y_pred)
    test_rmse = math.sqrt(test_mse)
    test_mae = mean_absolute_error(y_true, y_pred)
    test_explained_variance = explained_variance_score(y_true, y_pred)
#     test_r2 = r2_score(y_true, y_test_pred)

    print(prefix, "RMSE :", test_rmse)
    print(prefix, "MAE :", test_mae)
    print(prefix, "Explained Variance :", test_explained_variance)
#     print("Testing R2 :", test_r2)


def plot_prediction_wtih_pca(model, pca, X_train, X_test, y_train, y_test, name=None, alpha=0.2, save=True):
    if name is None:
        name = model.__class__.__name__

    X_train_reduced = pca.transform(X_train)
    X_test_reduced = pca.transform(X_test)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    val_metrics(y_train, y_train_pred, prefix='Training')
    val_metrics(y_test, y_test_pred)

    y_train_val_sorted_arg = np.argsort(y_train)
    y_test_sorted_arg = np.argsort(y_test)

    plt.figure(figsize=(15, 6))
    plt.plot(range(
        y_train.shape[0]), y_train.iloc[y_train_val_sorted_arg], label='Actual', color='b')
    plt.scatter(range(
        y_train.shape[0]), y_train_pred[y_train_val_sorted_arg], label='Predict', color='g', alpha=alpha)
    plt.legend()
    plt.title(name + ' model predictions (train)')
    plt.xlabel('Instance')
    plt.ylabel('People')
    if save is True:
        save_fig(name + '_pred_actual_instance' + '_train')
    plt.show()

    plt.figure(figsize=(15, 6))
    plt.plot(range(
        y_test.shape[0]), y_test.iloc[y_test_sorted_arg], label='Actual', color='b')
    plt.scatter(range(
        y_test.shape[0]), y_test_pred[y_test_sorted_arg], label='Predict', color='g', alpha=alpha)
    plt.legend()
    plt.title(name + ' model predictions (test)')
    plt.xlabel('Instance')
    plt.ylabel('People')
    if save is True:
        save_fig(name + '_pred_actual_instance' + '_test')
    plt.show()

    plt.figure(figsize=(15, 6))
    plt.scatter(X_train_reduced, y_train,
                label='Actual', color='b', alpha=alpha)
    plt.scatter(X_train_reduced, y_train_pred,
                label='Predict', color='g', alpha=alpha)
    plt.legend()
    plt.title(name + ' model predictions (train)')
    plt.xlabel('Reduced feature by PCA')
    plt.ylabel('People')
    if save is True:
        save_fig(name + '_pred_actual_reduced' + '_train')
    plt.show()

    plt.figure(figsize=(15, 6))
    plt.scatter(X_test_reduced, y_test,
                label='Actual', color='b', alpha=alpha)
    plt.scatter(X_test_reduced, y_test_pred,
                label='Predict', color='g', alpha=alpha)
    plt.legend()
    plt.title(name + ' model predictions (test)')
    plt.xlabel('Reduced feature by PCA')
    plt.ylabel('People')
    if save is True:
        save_fig(name + '_pred_actual_reduced' + '_test')
    plt.show()
