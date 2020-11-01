# sales_forecast
A web app for forecasting sales given historical data

# Table of contents
- [Purpose](#purpose)
- [Installation](#installation)
- [Usage](#usage)
- [Web Application](#web-application)
- [Project structure](#project-structure)
- [Data](#data)
- [Modelling](#modelling) 
- [Discussion of the results](#discussion-of-the-results)
- [Author](#author)
- [Credits](#credits)
- [Requirements](#requirements)


# Purpose
A web application that predicts sales volumes (number of items sold) next month based on historical data.
This potentaially could be interesting to small/medium business owners in retailing to help them plan their supply better.
Machine Learning has technologies for such task for a long time, but in many cases using them requires buying expensive software solustions or hiring a team of analytics.
A good forecasting tool would automize the tedious process of planning for the next month and also improve expert-based forecasts using data science.

# Installation
1. In order to install the code and deploy the app locally please download from Github: `git clone https://github.com/alibekU/pipsales_forecast`.
2. You may want to set up a new virtual environment: `python3 -m venv /path/to/new/virtual/environment` 
3. Then, use pip to install all the needed packages: `pip install -r requirements.txt`

# Usage
**To deploy the web app locally:**
After downloading, go to the the 'sales_forecast/app' folder and:
1. **From the app/ directory** run the following command to launch your web app.
    `python run.py`

2. Go to http://0.0.0.0:3001/

**To re-train the model:**
Currently the web app uses app/models/forecast_v1.pkl model that was pre-trained on 40 shops.
However, if you want to create a new model and re-train it run the following commands in the 'sales_forecast/app' directory. <br/>
Note that by default the model will be trained on 3 shops to save time.
Please note it will take 5-6 hours to train on 40 shops.
To run pipeline that cleans data and trains the model run this command from app/ directory
        `python train_model.py ../data/sales_train.csv ../data/items.csv ../models/forecast.pkl`


# Web Application
The app can be hosted locally only now.
![Web Application Interface](images/screenshot1.png)
<br/>
Instructions:
1. Download the template excel to see an example of input file structure (more on that below) 
2. Create your own excel with at least 6 month of sales data similarly to the template excel 
3. Upload your excel with data by pressing "Choose file" and locating the file 
4. Press 'Upload data and run prediction' button to generate a next month forecast
5. A link that allows to download results will apper after predicting is complete 

# Project structure 
data\
-sales_train.csv - data with sales transactions for 60 shops in 3 years from https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data
-items.csv - data with item categories for 60 shops in 3 years from https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data
models\
-forecast_v1.pkl - a Pickle file, saved regression model trained on 40 shops data
-forecast.pkl - a Pickle file, saved regression model trained on 3 shops data
app\
-run.py - the main script of Flask web app
-process_data.py - functions to clean, generate features and split data for predicting
-train_model.py - functions to train and evaluate a model. Run this file in command line to re-build a model
-templates\
 --master.html - main html page template
 --resulst.html - a template for displaying results
 -uploads\ - folder with data uploaded by the users
 -downloads\ - folder with data for users to download
images\ - pictures for the README file
requirements.txt - a list of required PIP packages, result of `pip freeeze` command
Procfile - code for Flask app launch at Heroku
data_exloration_v1.ipynb - a Jupyter notebook with ML pipeline exploration
README.md - readme file

# Data
The training data comes from Kaggle's Predict Future Sales competition https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data <br/>

Number of transactions is 1820364 <br/>
Number of shops is 40 <br/>
Number of categories is 74 <br/>
Number of unique items is 19111 <br/>

![plot2](images/screenshot2.png)
![plot3](images/screenshot3.png)
![plot4](images/screenshot4.png)

# Modelling
The model that was used for forecasting is XGBoost regressor. <br/>
In the code I currently use I've not performed grid search parameter optimization as it takes several our to train the model without it, but in hte future I plan to use less data and try to optimize the model.<br/>
In the modelling process I predict sales for the next month as the target variable. <br/>
I've added various feature that show how each shop, item and category have historically performed.
The most important features turned out to be category-grouped mean number of items sold next month, month-grouped mean number of items sold next month and shop-grouped mean number of items sold next month.
![plot5](images/screenshot4.png)

# Discussion of the results
Currently the mean absolute error (MAE) on number of items sold in a month is around 0.05 on test data that was derived from the shops in the train data. <br/>
However there are a lot of 0 values (items that were not sold in a month get assigned 0 value explicitly instead ob being omitted), and I have also measured MAE on non-zero actual or predicted values. For test data on shops that were used in training it is 1.63 items. Given mean number of items sold a month is 2.5 this is actually not such a small number.<br/>
When testing on shops that were not used in trainin, MAE is 0.07, MAE on non-zero count of items is 2.84. I will be working on imporoving these numbers.
Possible improvements: <br/>
Web app: <br/>
1. Interface
2. File check
3. Security
4. Possibly send result forecast files through email for privacy
5. Support multiple users work
<br/>
Model:<br/>
1. CV with time based split of train and test
2. Train on more shops

# Author 
- Alibek Utyubayev. 
<br/>
Linkedin: https://www.linkedin.com/in/alibek-utyubayev-74402721/

# Credits
Credits to Kaggle for the collected data and to Dimitreo Liveira for starting ideas on ML analysis https://www.kaggle.com/dimitreoliveira/model-stacking-feature-engineering-and-eda

# Requirements
click==7.1.2
Flask==1.1.2
gunicorn==20.0.4
itsdangerous==1.1.0
Jinja2==2.11.2
joblib==0.17.0
MarkupSafe==1.1.1
numpy==1.19.3
pandas==1.1.4
plotly==4.12.0
python-dateutil==2.8.1
pytz==2020.1
retrying==1.3.3
scikit-learn==0.23.2
scipy==1.5.3
six==1.15.0
sklearn==0.0
threadpoolctl==2.1.0
Werkzeug==1.0.1
xgboost==0.90
xlrd==1.2.0
xlwt==1.3.0