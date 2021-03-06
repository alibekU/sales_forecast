import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify, send_file, flash, redirect, url_for, send_from_directory
import plotly.graph_objs as Go
import numpy as np
import joblib
from process_data import load_data_excel, return_processed_data, create_prediction_df, save_data_excel
from train_model import evaluate_model, base_estimator
from werkzeug.utils import secure_filename
from xgboost import XGBRegressor
import os

UPLOAD_FOLDER = 'uploads/'
DOWLOAD_FOLDER = 'downloads/'
ALLOWED_EXTENSIONS = {'xls', 'xlsx','xml', 'xlsm'}


app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]iasdfffsd/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# index webpage displays cool visuals and receives user input text for model
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash('Wrong format')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('get_results', filename=filename))
    # render web page
    return render_template('master.html')

@app.route('/results/<filename>')
def get_results(filename):
    '''
        get_results() - function that processes user excel file, creates predictions and ouputs the results by rewriting original input file
        Input:
             filename - name of the user file with data
        Output:
            a web page with link to download the forecast in an excel file with the same name as the original upload file
    '''
    # get the location of a user data file
    file_location = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    # get the data into pandas dataframe
    data = load_data_excel(file_location)
    # process data
    X_train, Y_train, X_test, Y_test, X_predict, extended_test_set, extended_predict_set = return_processed_data(data)
    # load model
    model = joblib.load("../models/forecast_best_linear.pkl")

    # Compare with base estimator that returns last 3-month average of items sold
    estimator_last_3_months = base_estimator('last_three_months')

    # get the evaluation of prediction accuracy on test data from seen months
    mae, mae_base, error_better, cash_better, total_sales = evaluate_model(model, estimator_last_3_months, X_test, Y_test, extended_test_set)       

    min_month = extended_test_set['month'].iloc[0]
    min_year = extended_test_set['year'].iloc[0]
    max_month = extended_test_set['month'].iloc[-1]
    max_year = extended_test_set['year'].iloc[-1]

    date_min = '{}/{}'.format(min_month, min_year)
    date_max = '{}/{}'.format(max_month, max_year)
        
    if error_better >= 0:
        result_word1 = 'YES'
    else:
        result_word1 = 'NO'
    
    if error_better > 0:
        result_word2 = 'better'
    elif error_better == 0:
        result_word2 = ' different - meaning produced the same error'
    else:
        result_word2 = 'worse'

    saving = int(total_sales*cash_better)

    # predict the data for the next unseen month
    y_predict = model.predict(X_predict)
    # convert to int for meaningful forecast
    y_predict = y_predict.astype(int)

    # form an output pandas ds
    result_df = create_prediction_df(extended_predict_set, y_predict)

    # save this result dataframe into an excel file for the user to download
    save_data_excel(result_df, file_location)

    return render_template('results.html', 
                            filename=filename, 
                            mae=mae,
                            mae_base=mae_base, 
                            error_better=int(100*error_better),
                            cash_better=int(100*cash_better),
                            total_sales=total_sales,
                            date_min=date_min,
                            date_max=date_max,
                            result_word1=result_word1,
                            result_word2=result_word2,
                            saving=saving)

@app.route('/download_template')
def download_template():
    '''
        download_template() - function that lets users download template excel file
        Input:
            None
        Output:
            a file
    '''
    filename = 'excel_template.xls'
    file_location = os.path.join(app.config['DOWNLOAD_FOLDER'], filename)
    return send_file(file_location, as_attachment=True)

@app.route('/download/<filename>')
def download_file(filename):
    '''
        download_file() - function that lets users download result file from website's uploads folder
        Input:
            filename - (str) name to the needed file
        Output:
            a file
    '''
    filename = secure_filename(filename)
    file_location = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(file_location, as_attachment=True)


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()