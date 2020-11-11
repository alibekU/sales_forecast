'''
    process_data.py - ETL processes and creating of data and labels for ML pipeline for Sales Forecasting web app.
    Author: Alibek Utyubayev.
'''

# import libraries
import sys
import pandas as pd
import datetime
import math
import numpy as np


def preprocess_data(sales_filepath, items_filepath, use_shop_ids=[38,42,35,23,32,24,4,5,12,29]):
    '''
        preprocess_data() - function that transforms initial training data CSV files into a dataframe 
                            in a format that can be then processed and used in training pipeline
        Input:
            sales_filepath - (str) path to training sales data 
            items_filepath - (str) path to training data on items (names, categories)
            use_shop_ids - (list) a subset of int shop ids from 'sales_filepath' CSV file to use during training to reduce training time
        Output:
            data - (pd.DataFrame) a Pandas dataframe with sales data

    '''
    sales_train = pd.read_csv(sales_filepath)
    items = pd.read_csv(items_filepath)
    data = sales_train[sales_train['shop_id'].isin(use_shop_ids) ].drop(columns=['date_block_num']).reset_index(drop=True)
    data = pd.merge(data, items, on=['item_id'], how='left')    

    return data


def load_data_excel(data_filepath):
    '''
        load_data() - function that creates a Pandas dataframe from given excel file
        Input:
            data_filepath - (str) path to a excel file with sales data in needed format to train on or predict
            Needed columns are: date, shop_id, item_id, item_price, item_cnt_day, item_name, item_category_id
        Output:
            df - (pd.DataFrame) a Pandas dataframe with sales data
    '''
    return pd.read_excel(data_filepath)

def check_data_correctnes(data):
    '''
        check_data_correctnes() - function that checks correctness of a dataframe format and data, and assigns a shop_id if none is given
                                  in case of a single shop in the data
        Input:
            data - (pd.DataFrame) a Pandas dataframe with sales data
            Needed columns are: date, shop_id, item_id, item_price, item_cnt_day, item_name, item_category_id
        Output:
            data - (pd.DataFrame) a Pandas dataframe with sales data with necessary changes if needed (add shop_id if empty),
            OR
            raises an exception to let the web app know that format is incorrect 
    '''
    data['shop_id'] = data['shop_id'].fillna('shop1')
    return data

def clean_and_aggreagate(data):
    '''
        clean_and_aggreagate() - function that cleanes, aggregates and sorts a Pandas dataframe for training or predicting sales
        Input:
            data - (pd.DataFrame) a Pandas dataframe with sales data
        Output:
            data_monthly - (pd.DataFrame) a cleaned and aggregated sorted dataframe
    '''

    # delete data with negative item counts or prices
    data = data[data['item_cnt_day']>0]
    data = data[data['item_price']>0]
    # clean from any null values
    data = data.dropna()
    
    # convert date from string to datetime format
    data['date'] = pd.to_datetime(data['date'], format='%d.%m.%Y')

    # Add month and year and aggregate data
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year

    # making sure we are getting only needed columns
    keep_columns_agg = ['shop_id', 'item_id', 'item_price', 'item_cnt_day', 'month', 'year', 'item_category_id']
    
    # aggregate data monthly and by shops and items 
    data_monthly = data[keep_columns_agg].groupby(['year','month', 'shop_id', 'item_category_id', 'item_id'], as_index=False).agg(
        {'item_price':'mean', 
        'item_cnt_day':['sum', 'mean'] })

    # make multilevel index flat
    data_monthly.columns = data_monthly.columns.map(''.join)

    # rename columns
    data_monthly = data_monthly.rename(columns={
        'item_pricemean': 'item_price_avg', 
        'item_cnt_daysum':'item_cnt_month',
        'item_cnt_daymean':'item_cnt_day_mean'})
    
    # sort the dataframe for future analysis
    data_monthly = data_monthly.sort_values(by=['year', 'month', 'shop_id', 'item_id']).reset_index(drop=True)

    return data_monthly

def add_empty_rows(data_monthly):
    '''
        add_empty_rows() - function that extends a dataframe of aggregated sales with skipped rows
                           so that the dataframe has explicit data on zero sales (sets 'item_cnt_month' to zero) 
                           instead of not mentioning an item for a particular month in a particular shop
        Input:
            data_monthly - (pd.DataFrame) sorted Pandas dataframe with aggregated sales data
        Output:
            data_monthly_ext - (pd.DataFrame) extended dataframe with explicit 0 rows
    '''
    item_ids = data_monthly['item_id'].unique()
    shop_ids = data_monthly['shop_id'].unique()

    start_date_year = int(data_monthly.iloc[0,:]['year'])
    start_date_month = int(data_monthly.iloc[0,:]['month'])
    end_date_year = int(data_monthly.iloc[-1,:]['year'])
    end_date_month = int(data_monthly.iloc[-1,:]['month'])

    # --- calculate total number of months in the period of historical data
    number_of_months = (end_date_year - start_date_year) * 12 + (end_date_month - start_date_month) + 1

    # will have data on all possible combinations of sales records
    # for given shops, items and given time period 
    zero_rows = []
    cur_month = start_date_month
    cur_year = start_date_year

    for i in range(number_of_months):
        for shop in shop_ids:
            for item in item_ids:
                # adding a row - transaction for certain shop, year, month and item that will be zeroes initially
                # and then joined with main data so that main data has explicit zeros as transactions
                row = [cur_year, cur_month, shop, item]
                zero_rows.append(row)
        
        # create a correction coeeficient for the case of December, 12th month, so that we het 12 and not 0
        add_12_if_receive_0 = (12 - 12*math.ceil((cur_month+1)%12 / 12))
        # increase month by 1, make sure it will be in [1,12] range
        cur_month = (cur_month+1)%12 + add_12_if_receive_0
        
        # increase year if we arrived to the 1st month, January, after increasing month
        if cur_month == 1:
            cur_year += 1

    # create a df of zero rows    
    zero_rows = pd.DataFrame(zero_rows, columns=['year', 'month', 'shop_id', 'item_id'])

    # merge zero rows with the main data, data_monthly
    data_monthly_ext = pd.merge(zero_rows, data_monthly, on=['year','month', 'shop_id','item_id'], how='left')
    # fill missing records with 0s
    data_monthly_ext.fillna(0, inplace=True)
    # make sure the dataframe is sorted
    data_monthly_ext = data_monthly_ext.sort_values(by=['year', 'month', 'shop_id', 'item_id']).reset_index(drop=True)

    return data_monthly_ext

def add_global_features(data_monthly_ext):
    '''
        add_global_features() - function that adds features for training and predicting purposes.
                                I called features global becasue they can be applied to the whole data
                                before splitting into train and test without any data leakage in terms of forecasting
        Input:
            data_monthly_ext - (pd.DataFrame) a sorted by year and month Pandas dataframe with aggregated sales data
        Output:
            data_monthly_ext - (pd.DataFrame) initial dataframe with new features
    '''
    # --- add date_block to split data into train and test
    start_month = data_monthly_ext.iloc[0,:]['month']
    start_year = data_monthly_ext.iloc[0,:]['year']

    # order of the starting month - we will subtract it from other month to understand how many month we moved forward
    starting_month_agg = start_year*12 + start_month
    # create arrays of year and month to perform calculations on them using np.vectorize
    years = np.array(data_monthly_ext['year'])
    months = np.array(data_monthly_ext['month'])
    # for better performance create a vectorized function that computes block number (order of month from starting month)
    calculate_block_num = np.vectorize(lambda year, month: (year*12 + month)- starting_month_agg)
    # number of months since year 0 in starting date - this will be subtracted to understand how many months we have moved ahead from start
    data_monthly_ext['date_block_num'] = calculate_block_num(years, months)

    # Feature engineering
    # add rolling statistics

    # define size of a rolling window
    rolling_window_size = 3
    # --- Calculate rolling statistics
    def function_mean(column):
        return column.rolling(window=rolling_window_size, min_periods=1).mean()

    data_monthly_ext['item_cnt_roll_mean'] = data_monthly_ext.groupby(['shop_id','item_id'])['item_cnt_month'].apply(function_mean)
    
    # store average number of items sold per month for each item in a shop up to this point using pd.expanding function
    data_monthly_ext['shop_item_mean_past'] = data_monthly_ext.groupby(['shop_id', 'item_id'])[['item_cnt_month']].expanding().mean().values
    
    # store average number of items sold per month for each year up to this point using pd.expanding function
    data_monthly_ext['year_mean_past'] = data_monthly_ext.groupby(['year'])[['item_cnt_month']].expanding().mean().values
    
    # store average number of items sold per month for each month up to this point using pd.expanding function
    data_monthly_ext['month_mean_past'] = data_monthly_ext.groupby(['month'])[['item_cnt_month']].expanding().mean().values


    # let us calculate lags for the past 3 months
    number_lags = range(1,4)

    # --- generate shifted number of items sold from the past 1-3 months
    for lag in number_lags:
        feature_name = 'item_cnt_shifted{}'.format(lag)
        data_monthly_ext[feature_name] = data_monthly_ext.groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(lag)
        # Fill the empty shifted features with 0
        data_monthly_ext[feature_name] = data_monthly_ext[feature_name].fillna(0)

    # generate trend which shows the change in item sales count
    # trend = current - (previous_1 + ... + previous_n)/n = n*current - previous_1 - ... - previous_n, in our case n=3
    # initially fill with current sales multiplied by number of times we will substract previous values to get average
    data_monthly_ext['item_trend'] = len(number_lags) * data_monthly_ext['item_cnt_month']
    # then subtract previous n values
    for lag in number_lags:
        feature_name = 'item_cnt_shifted{}'.format(lag)
        data_monthly_ext['item_trend'] -= data_monthly_ext[feature_name]
    # then divide by the number of times we have subtracted previous values to get average
    data_monthly_ext['item_trend'] /= len(number_lags) 

    return data_monthly_ext


def add_labels(data_monthly_ext):
    '''
        add_labels() - function that creates labels = a column of values we will predict = sales next month for an item in a shop
        Input:
            data_monthly_ext - (pd.DataFrame) a Pandas dataframe with aggregated sorted sales data
        Output:
            data_monthly_ext - (pd.DataFrame) dataframe with new column - 'itm_cnt_nxt_mnth', which is sales next month for an item in a shop
    '''
    data_monthly_ext['itm_cnt_nxt_mnth'] = data_monthly_ext.groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(-1)
    return data_monthly_ext

def split_train_test_predict(data_monthly_ext):
    '''
        split_train_test_predict() - function that splits data into train, test and predict, where predict - is the last available month in the data,
                       and for which we will be generating prediction for the next, unknown to us month. Test will be used to see how good is the model.
                       Since we are dealing with timeseries, we cannot simply randomly split data. We also want to split exactly by month periods, so the standard 
                       split function will not work.
        Input:
            data_monthly - (pd.DataFrame) a Pandas dataframe with aggregated sales data
        Output:
            train_set - (pd.DataFrame) training data
            test_set - (pd.DataFrame) testing data to see how we performed
            predict_set - (pd.DataFrame) set of sales for the last month for which we will predict future sales for the next unseen month
    '''
    # calculate date_block_num for splitting
    num_months = data_monthly_ext['date_block_num'].max() + 1
    # starting from the 3rd month as we have rolling statistics with 3 month window (date_block_num starts with 0)
    train_low = 3 - 1
    # get approximately 70% of the data
    train_high = int(0.7 * num_months) - 1
    # testing data will have all the rest of the month up to a last one, which is used for predictiong as it does not have data on next month sales
    test_high = num_months - 1

    train_set = data_monthly_ext[(data_monthly_ext['date_block_num'] >= train_low) & (data_monthly_ext['date_block_num'] <= train_high )].copy()
    test_set = data_monthly_ext[(data_monthly_ext['date_block_num'] > train_high) & (data_monthly_ext['date_block_num'] < test_high )].copy()
    predict_set = data_monthly_ext[data_monthly_ext['date_block_num'] == test_high].copy()

    # in train and test get rid of the null values in target variable
    train_set = train_set.dropna(subset=['itm_cnt_nxt_mnth'])
    test_set = test_set.dropna(subset=['itm_cnt_nxt_mnth'])

    return train_set, test_set, predict_set



def add_set_features(train, test):
    '''
        add_set_features() - function for more feature engineering done on each train and test sets to avoid data leakage.
                            We will be calculating statistics based on the sales in the next month, so cannot use it on predict set as
                            it does not have information on the future sales. On test set we are setting values calculated based on train data
                            because we want to avoid data leakage and test data "knowing" about sales we want to predict.
        Input:
            train_set - (pd.DataFrame) training data
            test_set - (pd.DataFrame) testing data to see how we performed
        Output:
            train_set - (pd.DataFrame) training data with new features
            test_set - (pd.DataFrame) testing data with new features based on training data without data that can help forecast values on test,
                        only past values from train are added here
    '''
    # each new feature will be a statistics computed by grouping and aggregating next month sales by certain dimensions
    # we will generate that grouped statistics and then merge larger ungrouped train and test dataframes with it
    # computing on train data for both train and test as we don't want to give away any information to test set
    # as the calculations are done on future sale and since we are dealing with timeseries

    def generate_statistics(dataset, group_by_columns, new_column_names, agg_column='itm_cnt_nxt_mnth', agg_function_names=['mean']):
        '''
            generate_statistics() - helper internal function for generating statistics of a dataset using grouping and aggregation
            Input:
                dataset - (pd.DataFrame) data frame to calculate statistics on
                group_by_columns - (list of str) columns to group by
                new_column_names - (list of str) how to name new resulting columns
                agg_column - (str) on which column the calculation will be performed
                agg_function_names - (list of str) which aggregate functions to use on 'agg_column' after grouping

            Output:
                res - (pd.DataFrame) dataframe with 'new_column_names' columns - result of grouping and applying aggreagate functions
        '''
        res = dataset.groupby(group_by_columns).agg({agg_column: agg_function_names})
        res.columns = new_column_names
        res.reset_index(inplace=True)
        return res


    # --- Add mean statistic features to train and test sets.

    # Averages for each item across all shops, months and years
    item_means = generate_statistics(train, ['item_id'], ['item_mean_future'])
    train = pd.merge(train, item_means, on=['item_id'], how='left')
    test = pd.merge(test, item_means, on=['item_id'], how='left')

    # Averages for each item in each shop across all months and years
    shop_item_means = generate_statistics(train, ['shop_id', 'item_id'], ['shop_item_mean_future'])
    train = pd.merge(train, shop_item_means, on=['shop_id', 'item_id'], how='left')
    test = pd.merge(test, shop_item_means, on=['shop_id', 'item_id'], how='left')
    
    # Averages for each category across all shops, months and years
    category_means = generate_statistics(train, ['item_category_id'], ['category_mean_future'])
    train = pd.merge(train, category_means, on=['item_category_id'], how='left')
    test = pd.merge(test, category_means, on=['item_category_id'], how='left')
    
    # Averages for each month across all shops, years and items
    month_means = generate_statistics(train, ['month'], ['month_mean_future'])
    train = pd.merge(train, month_means, on=['month'], how='left')
    test = pd.merge(test, month_means, on=['month'], how='left')

    # Fill the empty features with 0
    train = train.fillna(0)
    test = test.fillna(0)

    return train, test


def split_data_labels(train_set, test_set, predict_set):
    '''
        split_data_labels() - function that splits sets into data (features) and labels to predict, 
            specifically train and test into X_train, Y_train, X_test, Y_test 
            and also creates X_predict for prediction in the same format as X_train and X_test.
            X_predict - is the last available month in the data, and for which we will be generating prediction for the next, unknown to us, month. 
            Test will be used to see how good is the model.
            Since we are dealing with timeseries, we cannot simply randomly split data. We also want to split exactly by month periods, so the standard 
            split function will not work.
        Input:
            train_set - (pd.DataFrame) training data
            test_set - (pd.DataFrame) testing data to see how we performed
            predict_set - (pd.DataFrame) set of sales for the last month for which we will predict future sales for the next unseen month
        Output:
            X_train - (pd.DataFrame) training features
            Y_train - (pd.DataFrame) training labels
            X_test - (pd.DataFrame) testing features
            Y_test - (pd.DataFrame) testing labels
            X_predict - (pd.DataFrame) features for predicting unknown data  
    '''
    # create train and test sets and labels. 
    X_train = train_set.drop(['itm_cnt_nxt_mnth'], axis=1)
    Y_train = train_set['itm_cnt_nxt_mnth'].astype(int)
    X_test = test_set.drop(['itm_cnt_nxt_mnth'], axis=1)
    Y_test = test_set['itm_cnt_nxt_mnth'].astype(int)
    
    # create X_predict to predct next unseen month
    history = pd.concat([train_set, test_set]).drop_duplicates(subset=['item_id'], keep='last')
    X_predict = pd.merge(predict_set, history, on=['item_id'], how='left', suffixes=['', '_'])
    X_predict.drop('itm_cnt_nxt_mnth', axis=1, inplace=True)
    X_predict = X_predict[X_train.columns]

    return X_train, Y_train, X_test, Y_test, X_predict


def save_data_csv(df,filepath):
    '''
        save_data() - a function that saves a Pandas dataframe into a CSV file for later training
                    Not used on new data from web app as that data will be only used for testing and predicting using existing model
        Input:
            df -  a Pandas dataframe with data to save
            filepath - a path to a CSV file where to save the data
        Output:
            None
    '''
    df.to_csv(filepath, index=False)

    return None

def save_data_excel(df,filepath):
    '''
        save_data() - a function that saves a Pandas dataframe into an excel file.
        Input:
            df -  a Pandas dataframe with data to save
            filepath - a path to an excel file where to save the data
        Output:
            None
    '''
    df.to_excel(filepath, index=False)

    return None
    

def return_processed_data(data):
    '''
        return_processed_data() - function that combines all of the ETL steps for ML training and predicting.
                                  Raises exception if datafile is not in needed format or has less than 6 month of data
        Input:
            data - (pd.DataFrame) dataframe with sales data.
                    Needed columns are: date, shop_id, item_id, item_price, item_cnt_day, item_name, item_category_id
        Output:
            X_train - (pd.DataFrame) training features
            Y_train - (pd.DataFrame) training labels
            X_test - (pd.DataFrame) testing features
            Y_test - (pd.DataFrame) testing labels
            X_predict - (pd.DataFrame) data set (features) for predicting unknown data
            extended_test_set - (pd.DataFrame) data set of test data features, like X_test, but with all the original columns (like item_id and etc.) to return to a user later
            extended_predict_set - (pd.DataFrame) data set for predicting data, like X_predict, but with all the original columns (like item_id and etc.) to return to a user later
    '''
    data = check_data_correctnes(data)
    data_monthly = clean_and_aggreagate(data)
    data_monthly_ext = add_empty_rows(data_monthly)
    data_monthly_ext = add_global_features(data_monthly_ext)
    data_monthly_ext = add_labels(data_monthly_ext)
    train_set, test_set, predict_set = split_train_test_predict(data_monthly_ext)
    train_set, test_set = add_set_features(train_set, test_set)
    X_train, Y_train, X_test, Y_test, X_predict = split_data_labels(train_set, test_set, predict_set)


    # --- select features that will be used for training, testing and predicting
    features = ['shop_item_mean_future', 'item_cnt_month', 'item_trend', 'month_mean_future', 'item_cnt_roll_mean', 'item_cnt_day_mean', 
                'month_mean_past', 'item_mean_future', 'category_mean_future', 'shop_item_mean_past', 'year_mean_past']
    # ---- Select subsets
    # First, save the data we will be predicting before we will select features for modelling from it - 'extended_predict_set'. 
    # This will allow us to return prediction in a clear format to the user
    extended_test_set = X_test
    extended_predict_set = X_predict
    # --- subset other Xs            
    X_train = X_train[features]
    X_test = X_test[features]
    X_predict = X_predict[features]

    return X_train, Y_train, X_test, Y_test, X_predict, extended_test_set, extended_predict_set

def create_prediction_df(extended_predict_set, Y_predict, columns = ['year', 'month', 'shop_id', 'item_id', 'item_category_id', 'item_price_avg']):
    '''
        create_prediction_df() - a function that combines predicted target data (sales amount) with the data itself (shops, items, categories) into one Pandas dataframe
        Input:
            extended_predict_set - (pd.dataframe) dataframe with data (year, month, shops, items, categories, etc.)
            Y_predict - (list or numpy array) dataframe with predicted sales count for the next month
            columns = columns of extended_predict_set to keep
        Output:
            result_df - resulting df with combined data
    '''    
    result_df = extended_predict_set[columns].copy()

    # get current month and year
    cur_month = extended_predict_set['month'].iloc[0]
    cur_year = extended_predict_set['year'].iloc[0]

    # increase current month by 1 to output prediction
    cur_month += 1
    # check that we are less than 12, set to 1 if 13. '%' alone won't work as there is case of 12 that gives 0
    if cur_month == 13:
        cur_month = 1
        # increae a year by one
        cur_year += 1

    result_df.loc[:,'year'] = cur_year
    result_df.loc[:,'month'] = cur_month

    # add predicted sales
    result_df.loc[:,'predicted_number_items_sold'] =  Y_predict

    return result_df