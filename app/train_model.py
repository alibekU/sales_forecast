'''
    train_model.py - ML pipeline of the Sales Forecast web app.
                          Functions are used to: 
                          1. train the model on data once initially using command line execution 
                          2. use created model in the web app. Only evaluate_model() function is called from the web app
    Author: Alibek Utyubayev.
    Usage:
        Need to pass following arguments as sys.argv to the program to train the model and run this file using python:
            sales_filepath - (str) path to training sales data 
            items_filepath - (str) path to training data on items (names, categories)
            model_filepath - a string with a filepath to a Pickle file where ML model that was trained on the data
'''

# import libraries
import sys
import pandas as pd
from sklearn.pipeline import Pipeline
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from xgboost import plot_importance
from process_data import preprocess_data, return_processed_data
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import TimeSeriesSplit


def load_data_csv(filepath):
    '''
        load_data() - function that creates a Pandas dataframe by reading data from CSV file
        Input:
            filepath - (str) a filepath to a CSV file where the data is stored
        Output:
            data - (pd.DataFrame) dataframe with data
    '''
    data = pd.read_csv(filepath)
    return data 


def build_model():
    '''
        build_model() - function that creates a model to later train, test and save for predicting
        Input:
            None 
        Output:
            model - a sklearn GridSearchCV object, model to be trained on existing data and predict categories for the new data
    '''

    
    linear_pipeline = Pipeline(steps=[
                    ('Scaler', StandardScaler()),
                    ('Regressor', HuberRegressor(max_iter=300, epsilon=1.3))
                  ])

    # To save time the gridsearch cv is commented out as I've already ran it before and inputted parameters above
    '''
    parameters = {
                    'Regressor__tol': [0.0001, 0.0005]
                    'Regressor__max_iter': [300, 400]
                    'Regressor__epsilon': [1.3, 1.35]
                 }

    cv_linear = GridSearchCV(
                    linear_pipeline, 
                    param_grid = parameters, 
                    n_jobs=-1, 
                    cv=TimeSeriesSplit(n_splits=3), 
                    scoring='neg_mean_absolute_error')
    model = cv_linear
    '''
    model = linear_pipeline
    return model


def evaluate_model(model, base_estimator, X_test, y_test, extended_test_set):
    '''
        evaluate_model() - function that evaluates an sklearn model
        Input:
            model - a trained sklearn model capable of  'predict' methods
            X_test - (pd.DataFrame) data for testing, features
            y_test - (np.array) array with labels for testing, targets
        Output:
            MAE - (float) - mean absolute error across all shops and items montly data for period of time defined in X_test dataset
    '''
    
    y_test = np.array(y_test)
    # get the model prediction
    y_pred = model.predict(X_test)
    # make integers for meaningfull predictions
    y_pred = y_pred.astype(int)
    MAE = mean_absolute_error(y_test, y_pred)
    
    # get the base estimator prediction
    y_pred_base = base_estimator.predict(X_test)
    # make integers for meaningfull predictions
    y_pred_base = y_pred_base.astype(int)
    MAE_base = mean_absolute_error(y_test, y_pred_base)
    
    error_diff = (MAE_base - MAE)
    
    if MAE_base != 0:
        error_better = error_diff/MAE_base
    elif error_diff == 0:
        error_better = 0
    else:
        error_better =  None    
    
    prices = np.array(extended_test_set['item_price_avg'])
    total_sales = np.sum(prices*y_test)
            
    diff_from_fact = np.abs(y_pred - y_test)
    diff_from_fact_base = np.abs(y_pred_base - y_test)
    
    cash_error_model = np.sum(diff_from_fact*prices)
    cash_error_base = np.sum(diff_from_fact_base*prices)
    
    cash_diff = (cash_error_base - cash_error_model)
    
    if total_sales != 0:
        cash_better = cash_diff/total_sales
    elif cash_diff == 0:
        cash_better = 0
    else:
        cash_better = None
        
    return round(MAE,2), round(MAE_base,2) , round(error_better,2), round(cash_better,2), int(total_sales)


def save_model(model, model_filepath):
    '''
        save_model() - function that saves a classification model into a Pickle file for later use in web app
        Input:
            model -  a sklearn Pipeline object - model that was trained on existing data and can predict categories for the new data
            model_filepath - a string with a filepath to a Pickle file where ML model, that was trained on the data and is ready to classify new messages, will be stored
        Output:
            None
    '''
    file_pkl = open(model_filepath, 'wb')
    pickle.dump(model, file_pkl)
    file_pkl.close()
    return None

class base_estimator:
    '''
        base_estimator - class for returning basic prediction for items sold next month 
                        to compare with ML estimators
        Parameters:
            type_ - (str) type of base estimator, either 'last_month' (by default) which returns number of items sold this month,
                    or 'last_three_months' which returns average of number of items sold last 3 months 
                    as the prediction of items sold next month
    '''
    def __init__(self, type_='last_month'):
        self.type_ = type_
    
    def predict(self, X_test):
        '''
            predict() - method that returns predictions for the nest month items sold
            Input:
                X_test - (pd.DataFrame) dataframe with contains number of items sold in the past and this month
            Output:
                y_pred - (np.array) array of predictions for the items sold next month
        '''
        if self.type_ == 'last_month':
            y_pred = np.array(X_test['item_cnt_month'])
        elif self.type_ == 'last_three_months':
            y_pred = np.array(X_test['item_cnt_roll_mean'])
        return y_pred


def main():
    '''
        main() - function that controls ML pipeline
        Input:
            None the function, but need input as system arguments to the program
        Output:
            None
    '''
    if len(sys.argv) == 4:
        sales_filepath, items_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n ')
        data = preprocess_data(sales_filepath, items_filepath, use_shop_ids=range(3))

        print('Cleaning and transforming data...')
        X_train, Y_train, X_test, Y_test, X_predict, extended_test_set, extended_predict_set = return_processed_data(data)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...') 

        # Compare with base estimator that returns last 3-month average of items sold
        estimator_last_3_months = base_estimator('last_three_months')

        mae, mae_base, error_better, cash_better, total_sales = evaluate_model(model, estimator_last_3_months, X_test, Y_test, extended_test_set)       

        min_month = extended_test_set['month'].iloc[0]
        min_year = extended_test_set['year'].iloc[0]
        max_month = extended_test_set['month'].iloc[-1]
        max_year = extended_test_set['year'].iloc[-1]
        
        if error_better >= 0:
            result_word = 'YES'
        else:
            result_word = 'NO'
        
        print('Can you trust the given forecast? Probably {}.'.format(result_word))
        print('To answer that question we checked the forecast-genereating model on test period: from {}/{} to the end of {}/{}.'.format(min_month, min_year, max_month, max_year))
        print('Then, the model was compared to a simple estimator which forecasts items sold next month as the average of items sold last 3 months.')
        print('Mean average error for the model on test data is {} items.'.format(mae) )
        print('Mean average error for the simple estimator on test data is {} items.'.format(mae_base) )
        
        if error_better > 0:
            result_word = 'better'
        elif error_better == 0:
            result_word = ' different - meaning produced the same error'
        else:
            result_word = 'worse'
        
        print('In terms of the number of items sold, the model was {}% {} compared to the simple estimator.'.format(round(100*error_better,2), result_word))
        saving = int(total_sales*cash_better)
        if cash_better > 0:
            print('In terms of the money, the model could have helped to better allocate {} or {}% of total sales of {} during the test period.*'.format(saving, round(100*cash_better,2), total_sales))
            print('   *These figures are calculated by comparing model forecast with the simple estimator and could have been achivied by stocking items that customers would purchase and not stocking items that were not ultimately purchased.')

        print('Model with these parameters turned out to be the best:', model.get_params())

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        print('Trained model saved!')

    else:
        print('Please provide 3 arguments:'\
                '1. filepath of the sales dataset items '\
                '2. filepath of the items dataset items'\
                '3. filepath of the pickle file to '\
              'save the model to as an argument to program. \n\nExample:'\
              'python train_model.py ../data/sales_train.csv ../data/items.csv ../models/forecast.pkl')


if __name__ == '__main__':
    main()