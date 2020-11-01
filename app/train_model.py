'''
    train_classifier.py - ML pipeline of the Sales Forecast web app.
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

    '''
    parameters = {
                "max_depth"        : [ 1, 3],
                "min_child_weight" : [ 7, 10]
    }
    '''        
    # using GridSearch for optimization is not a good idea for time series since we can only test on future data
    #cv_xgb = GridSearchCV(XGBRegressor(), param_grid = parameters, n_jobs=-1, cv=3)
    #model = cv_xgb
    model = XGBRegressor()
    return model


def evaluate_model(model, X_test, y_test):
    '''
        evaluate_model() - function that evaluates an sklearn model
        Input:
            model - a trained sklearn model capable of  'predict' methods
            X_test - data for testing, features
            y_test - labels for testing, targets
        Output:
            MAE - (float) - mean absolute error across all shops and items montly data for period of time defined in X_test dataset
    '''
    # get the model prediction
    y_pred = model.predict(X_test)
    # make integers for meaningfull predictions
    y_pred = y_pred.astype(int)
    MAE = mean_absolute_error(y_test, y_pred)

    validate_act_vs_pred = pd.DataFrame(zip(y_test, y_pred), columns=['actual', 'prediction'])
    validate_non_zero = validate_act_vs_pred[(validate_act_vs_pred['actual'] != 0) | (validate_act_vs_pred['prediction'] != 0)]
    MAE_non_zero = mean_absolute_error(validate_non_zero['actual'], validate_non_zero['prediction'])
    return MAE, MAE_non_zero


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
        data = preprocess_data(sales_filepath, items_filepath, use_shop_ids=[38,42,7])

        print('Cleaning and transforming data...')
        X_train, Y_train, X_test, Y_test, X_predict, extended_predict_set = return_processed_data(data)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        mae, mae_non_zero = evaluate_model(model, X_test, Y_test)
        print('Evaluating model...')
        print('MAE on test data is {}, mae non-zero on test data is {}'.format(mae, mae_non_zero))
        print('XGBRegressor with this parameters turned out to be the best:', model.best_estimator_.get_params())

        print(model.best_params_)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide 3 arguments:'\
                '1. filepath of the sales dataset items '\
                '2. filepath of the items dataset items'\
                '3. filepath of the pickle file to '\
              'save the model to as an argument to program. \n\nExample: python '\
              'train_classifier.py ../data/train_sales.csv ../data/items.csv  ../model/forecast.pkl')


if __name__ == '__main__':
    main()