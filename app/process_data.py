'''
    process_data.py - ETL processes for Sales Forecasting web app.
    
    Author: Alibek Utyubayev.

    Usage:
        Need to pass following arguments as sys.argv to the program:
            
'''

# import libraries
import pandas as pd
from sqlalchemy import create_engine
import sys
import numpy as np
import scipy.stats as ss


def preprocess_data(sales_filepath, items_filepath, new_file_name, use_shop_ids=[38,42,35,23,32,24,4,5,12,29]):
    '''
        preprocess_data() - function that transforms initial training data into a dataframe and saves in a CSV file 
                            in a format that can be then processed and used in training pipeline, just like data for predicting later
        Input:
            sales_filepath - (str) path to training sales data 
            items_filepath - (str) path to training data on items (names, categories)
            new_file_name - (str) path and name of the CSV file where the data will be stored
            use_shop_ids - (list) a subset of int shop ids from 'sales_filepath' CSV file to use during training to reduce training time
        Output:
            None, as dataframe is saved in a CSV file for later use

    '''
    sales_train = pd.read_csv(sales_filepath)
    items = pd.read_csv(items_filepath)
    data = sales_train[sales_train['shop_id'].isin(use_shop_ids) ].drop(columns=['date_block_num']).reset_index(drop=True)
    data = pd.merge(data, items, on=['item_id'], how='left')    

    data.to_csv(new_file_name, index=False)


def load_data(data_filepath):
    '''
        load_data() - function that creates a Pandas dataframe from given CSV files
        Input:
            data_filepath - (str) path to a CSV file with sales data in needed format to train on or predict
            Needed columns are: date, shop_id, item_id, item_price, item_cnt_day, item_name, item_category_id
        Output:
            df - (pd.DataFrame) a Pandas dataframe with sales data
    '''
    return pd.read_csv(data_filepath)

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
    pass

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
    data['date'] = pd.to_datetime(data['date']) #format='%d.%m.%Y'

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
            data_monthly - (pd.DataFrame) a Pandas dataframe with aggregated sales data
        Output:
            data_monthly_ext - (pd.DataFrame) extended dataframe with explicit 0 rows
    '''
    item_ids = data_monthly['item_id'].unique()
    shop_ids = data_monthly['shop_id'].unique()

    start_date = data['date'].min()
    end_date = data['date'].max()

    start_month = start_date.month
    start_year = start_date.year

    # calculate total number of months in the period of historical data
    number_of_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1

    # will have data on all possible combinations of sales records
    # for given shops, items and given time period 
    empty_df = []
    cur_month = start_month
    cur_year = start_year

    for i in range(number_of_months):
        for shop in shop_ids:
            for item in item_ids:
                empty_df.append([cur_month, cur_year, shop, item])
        
        add_12_if_receive_0 = (12 - 12*math.ceil((cur_month+1)%12 / 12))
        cur_month = (cur_month+1)%12 + add_12_if_receive_0
        
        if cur_month == 1:
            cur_year += 1
        
    empty_df = pd.DataFrame(empty_df, columns=['year', 'month', 'shop_id', 'item_id'])

    data_monthly_ext = pd.merge(empty_df, data_monthly, on=['year','month', 'shop_id','item_id'], how='left')
    # missing records will be filled with 0s
    data_monthly_ext.fillna(0, inplace=True)
    # make sure the dataframe is sorted
    data_monthly_ext = data_monthly_ext.sort_values(by=['year', 'month', 'shop_id', 'item_id']).reset_index(drop=True)

    return data_monthly_ext

def add_global_features(data_monthly_ext)
     '''
        add_global_features() - function that adds features for training and predicting purposes.
                                I called features global becasue they can be applied to the whole data
                                before splitting into train and test without any data leakage in terms of forecasting
        Input:
            data_monthly_ext - (pd.DataFrame) a sorted by year and month Pandas dataframe with aggregated sales data
        Output:
            data_monthly_ext - (pd.DataFrame) initial dataframe with new features
    '''
    # add date_block to split data into train and test
    start_month = data_monthly_ext.iloc[0,:]['month']
    start_year = data_monthly_ext.iloc[0,:]['year']
    # number of months since year 0 in starting date - this will be subtracted to understand how many months we have moved ahead from start
    starting_month_agg = start_year*12 + start_month
    years = np.array(data_monthly_ext['year'])
    months = np.array(data_monthly_ext['month'])
    calculate_block_num = np.vectorize(lambda year, month: (year*12 + month)- starting_month)
    data_monthly_ext['date_block_num'] = calculate_block_num(years, months)

    # Feature engineering
    # add rolling statistics

    rolling_window_size = 3
    # Min value
    f_min = lambda column: column.rolling(window=rolling_window_size, min_periods=1).min()
    # Max value
    f_max = lambda column: column.rolling(window=rolling_window_size, min_periods=1).max()
    # Mean value
    f_mean = lambda column: column.rolling(window=rolling_window_size, min_periods=1).mean()
    # Standard deviation
    f_std = lambda column: column.rolling(window=rolling_window_size, min_periods=1).std()
    ​
    # keep functions in a list to iterate
    functions = [f_min, f_max, f_mean, f_std]
    # these are suffixes to add to column names to generate new names
    suffixes = ['min', 'max', 'mean', 'std']
    ​
    # create len(functions) new features
    for i in range(len(functions)):
        data_monthly_ext['item_cnt_roll_{}'.format(suffixes[i])] = data_monthly_ext.groupby(['shop_id','item_id'])['item_cnt_month'].apply(functions[i])
    ​
    # Fill the empty std features with 0
    data_monthly_ext['item_cnt_roll_std'].fillna(0, inplace=True)

    # store average number of items sold per month for each item in a shop up to this point using pd.expanding function
    data_monthly_ext['item_mean_past'] = data_monthly_ext.groupby(['shop_id', 'item_category_id'])[['item_cnt_month']].expanding().mean().values

    # store average number of items sold per month for each category in a shop up to this point using pd.expanding function
    data_monthly_ext['category_mean_past'] = data_monthly_ext.groupby(['shop_id', 'item_id'])[['item_cnt_month']].expanding().mean().values


    lag_list = [1, 2, 3]

    # generate shifted number of items sold from the past 1-3 months
    for lag in lag_list:
        feature_name = 'item_cnt_shifted{}'.format(lag)
        data_monthly_ext[feature_name] = data_monthly_ext.sort_values('date_block_num').groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(lag)
        # Fill the empty shifted features with 0
        data_monthly_ext[feature_name].fillna(0, inplace=True)

    # generate trend which shows the change in item sales count
    # trend = current - (previous_1 + ... + previous_n)/n = n*current - previous_1 - ... - previous_n, in our case n=3
    # initially fill with current sales multiplied by number of times we will substract previous values to get average
    data_monthly_ext['item_trend'] = len(lag_list) * data_monthly_ext['item_cnt_month']
    # then subtract previous n values
    for lag in lag_list:
        feature_name = 'item_cnt_shifted{}'.format(lag)
        data_monthly_ext['item_trend'] -= data_monthly_ext[feature_name]
    # then divide by the number of times we have subtracted previous values to get average
    data_monthly_ext['item_trend'] /= len(lag_list) 

    return data_monthly_ext

def create_labels(data_monthly_ext)
     '''
        create_labels() - function that creates labels = a column of values we will predict = sales next month for an item in a shop
        Input:
            data_monthly_ext - (pd.DataFrame) a Pandas dataframe with aggregated sorted sales data
        Output:
            data_monthly_ext - (pd.DataFrame) dataframe with new column - 'itm_cnt_nxt_mnth', which is sales next month for an item in a shop
    '''
    data_monthly_ext['itm_cnt_nxt_mnth'] = data_monthly_ext.groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(-1)
    return data_monthly_ext

def split_data_train_test(data_monthly_ext)
     '''
        split_data_train_test() - function that splits data into train, test and predict, where predict - is the last available month in the data,
                       and for which we will be generating prediction for the next, unknown to us month. Test will be used to see how good is the model
        Input:
            data_monthly - (pd.DataFrame) a Pandas dataframe with aggregated sales data
        Output:
            train_set - (pd.DataFrame) training data
            test_set - (pd.DataFrame) testing data to see how we performed
            predict_set - (pd.DataFrame) set of sales for the last month for which we will predict future sales for the next unseen month
    '''
    # calculate date_block_num for splitting
    num_months = data_monthly_ext['date_block_num'].max() + 1
    # starting from the 3rd month as we have rolling statistics with 3 month window
    train_low = 3
    # get approximately 70% of the data
    train_high = int(0.7 * num_months)
    # testing data will have all the rest of the month up to a last one, which is used for predictiong as it does not have data on next month sales
    test_high = num_months - 1

    train_set = data_monthly_ext.query('date_block_num >= @train_low and date_block_num <= @train_high').copy()
    test_set = data_monthly_ext.query('date_block_num > @train_high and date_block_num < @test_high').copy()
    predict_set = data_monthly_ext.query('date_block_num == @test_high').copy()

    train_set.dropna(subset=['itm_cnt_nxt_mnth'], inplace=True)
    test_set.dropna(subset=['itm_cnt_nxt_mnth'], inplace=True)

    train_set.dropna(inplace=True)
    test_set.dropna(inplace=True)

    return train_set, test_set, predict_set
    '''
    print('Train set records:', train_set.shape[0])
    print('Validation set records:', validation_set.shape[0])
    print('Test set records:', test_set.shape[0])

    print('Train set records: %s (%.f%% of complete data)' % (train_set.shape[0], ((train_set.shape[0]/data_monthly_ext.shape[0])*100)))
    print('Validation set records: %s (%.f%% of complete data)' % (validation_set.shape[0], ((validation_set.shape[0]/data_monthly_ext.shape[0])*100)))
    '''

def generate_global_statistics(dataset, group_by_columns, new_column_name, agg_column='itm_cnt_nxt_mnth', agg_function_name='mean'):
     '''
        add_set_features() - function for generating statistics of a dataset using grouping and aggregation
        Input:
            dataset - (pd.DataFrame) data frame to calculate statistics on
            group_by_columns - (list of str) testing data to see how we performed
            new_column_name -
            agg_column -
            agg_function_name -

        Output:
            train_set - (pd.DataFrame) training data with new features
            test_set - (pd.DataFrame) testing data with new features based on training data without data that can help forecast values on test,
                        only past values from train are added here
    '''
    res = dataset.groupby(group_by_column).agg({agg_column: agg_function_name})
    res.columns = [new_column_name]
    res.reset_index(inplace=True)
    return res


def add_set_features(train, test)
     '''
        add_set_features() - function for more feature engineering done on each train and test sets to avoid data leakage
        Input:
            train_set - (pd.DataFrame) training data
            test_set - (pd.DataFrame) testing data to see how we performed
        Output:
            train_set - (pd.DataFrame) training data with new features
            test_set - (pd.DataFrame) testing data with new features based on training data without data that can help forecast values on test,
                        only past values from train are added here
    '''
    # Item mean
    gp_item_mean = train_set.groupby(['item_id']).agg({'itm_cnt_nxt_mnth': ['mean']})
    gp_item_mean.columns = ['item_mean_future']
    gp_item_mean.reset_index(inplace=True)
    # Year mean
    gp_year_mean = train_set.groupby(['year']).agg({'itm_cnt_nxt_mnth': ['mean']})
    gp_year_mean.columns = ['year_mean_future']
    gp_year_mean.reset_index(inplace=True)
    # Month mean
    gp_month_mean = train_set.groupby(['month']).agg({'itm_cnt_nxt_mnth': ['mean']})
    gp_month_mean.columns = ['month_mean_future']
    gp_month_mean.reset_index(inplace=True)
    # Category mean
    gp_category_mean = train_set.groupby(['item_category_id']).agg({'itm_cnt_nxt_mnth': ['mean']})
    gp_category_mean.columns = ['category_mean_future']
    gp_category_mean.reset_index(inplace=True)
    # Shop mean
    gp_shop_mean = train_set.groupby(['shop_id']).agg({'item_cnt_month': ['mean']})
    gp_shop_mean.columns = ['shop_mean_future']
    gp_shop_mean.reset_index(inplace=True)
    # Shop with item mean
    gp_shop_item_mean = train_set.groupby(['shop_id', 'item_id']).agg({'item_cnt_month': ['mean']})
    gp_shop_item_mean.columns = ['shop_item_mean_future']
    gp_shop_item_mean.reset_index(inplace=True)


    # Add mean encoding features to train set.
    train_set = pd.merge(train_set, gp_item_mean, on=['item_id'], how='left')
    train_set = pd.merge(train_set, gp_year_mean, on=['year'], how='left')
    train_set = pd.merge(train_set, gp_month_mean, on=['month'], how='left')
    train_set = pd.merge(train_set, gp_category_mean, on=['item_category_id'], how='left')
    train_set = pd.merge(train_set, gp_shop_mean, on=['shop_id'], how='left')
    train_set = pd.merge(train_set, gp_shop_item_mean, on=['shop_id', 'item_id'], how='left')

    # Add mean encoding features to validation set.
    test_set = pd.merge(test_set, gp_item_mean, on=['item_id'], how='left')
    test_set = pd.merge(test_set, gp_year_mean, on=['year'], how='left')
    test_set = pd.merge(test_set, gp_month_mean, on=['month'], how='left')
    test_set = pd.merge(test_set, gp_category_mean, on=['item_category_id'], how='left')
    test_set = pd.merge(test_set, gp_shop_mean, on=['shop_id'], how='left')
    test_set = pd.merge(test_set, gp_shop_item_mean, on=['shop_id', 'item_id'], how='left')

    return train_set, test_set



def create_predict(train, test)
     '''
        add_empty_rows() - function that extends a dataframe of aggregated sales with skipped rows
                           so that the dataframe has explicit data on zero sales (sets item_cnt_month to zero) 
                           instead of not mentioning an item for a particular month in a particular shop
        Input:
            data_monthly - (pd.DataFrame) a Pandas dataframe with aggregated sales data
        Output:
            data_monthly_ext - (pd.DataFrame) extended dataframe with explicit 0 rows
    '''


def return_processed_data(data_filepath)
     '''
        add_empty_rows() - function that extends a dataframe of aggregated sales with skipped rows
                           so that the dataframe has explicit data on zero sales (sets item_cnt_month to zero) 
                           instead of not mentioning an item for a particular month in a particular shop
        Input:
            data_monthly - (pd.DataFrame) a Pandas dataframe with aggregated sales data
        Output:
            data_monthly_ext - (pd.DataFrame) extended dataframe with explicit 0 rows
    '''

def save_data(df, table_name, database_filename):
    '''
        save_data() - a function that saves a Pandas dataframe into sqlite database
        Input:
            df -  a Pandas dataframe with data to save
            table_name - a string with the name of the table where df will be stored
            database_filename - a string with a filepath to a sqllite database file where data should be stored. If it does not exist, then a new one will be created 
            need_index - a boolean, False by default, if True - saves the index in the DB as well
        Output:
            None
    '''



def main():
    '''
        main() - function that performs an ETL process on messages and categories data 
                and saves data for data visuals in DB for future use by the web app
        Input:
            None the function, but need input as system arguments to the program
        Output:
            None
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        # get the messages and categories from CSV files
        df = load_data(messages_filepath, categories_filepath)

        # transform the messages and categories data
        print('Cleaning data...')
        df = clean_data(df)
        
        # load the messages and categories data
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, "Messages", database_filepath)

        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'Disaster_response.db')


if __name__ == '__main__':
    main()