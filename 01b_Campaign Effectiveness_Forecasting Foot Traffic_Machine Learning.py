# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/campaign-effectiveness. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/sales-forecasting

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Identifying Campaign Effectiveness For Forecasting Foot Traffic: Machine Learning
# MAGIC <div >
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2020/10/ds-architecture.png" style="width:900px;height:450px;">
# MAGIC </div>

# COMMAND ----------

# MAGIC %run ./_resources/00-setup-ml

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### In this notebook, we are going to look closely at foot traffic in New York City (the curated dataset) to understand how Subway Restaurants' various advertising campaign efforts drive the in-store visits. The goal of this workflow is 
# MAGIC ###### 1. To create a machine learning approach that, given a set of new campaign ads effort, predicts the in-store number of visits 
# MAGIC ###### 2. Leverage SHAP model interpreter, given a time series foot traffic in-store visits, quantify how much they were driven by a certain media channel.
# MAGIC 
# MAGIC #### The main steps taken in this notebook are: 
# MAGIC 
# MAGIC * Read in exploded versions of the curated Gold table which contains the NYC area Subway Restaurant daily time series Safegraphe foot traffic
# MAGIC * Exploratory analysis of features: distribution check, variable transformation
# MAGIC * Xgboost model to predict number of store visits: modern attribution model with Databricks mlflow, HyperOpt, AutoML
# MAGIC * Ascertaining the effectiveness of each Media input on the Foot Traffic. Make a conclusion on whether or not different media channels campaign were effective
# MAGIC * Use SHAP to interpret the attribution of each media channel for the in store Foot Traffic, and make actionable insight/ recommendation of media spend optimization

# COMMAND ----------

# MAGIC %md ### Load full foot traffic data

# COMMAND ----------

# DBTITLE 1,Use Subway Subset Foot Traffic Gold Table
# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE footTrafficMedia
# MAGIC SHALLOW CLONE footTrafficGold 

# COMMAND ----------

# MAGIC %md ### Distribution of our target feature num_visits for NY State 
# MAGIC ##### Exploratory analysis with Silver Delta Table
# MAGIC 
# MAGIC ##### Looks like we have multimodal distribution here (2 peaks). This may due to underlying differences of population

# COMMAND ----------

# DBTITLE 0,Distribution num_visits
# MAGIC %sql
# MAGIC select * from 
# MAGIC (select region, city, cast(year as integer) year, cast(month as integer) month, cast(day as integer) day, sum(num_visits) num_visits 
# MAGIC from subway_foot_traffic 
# MAGIC where region = 'NY' and num_visits >= 50
# MAGIC group by region, city, cast(year as integer), cast(month as integer), cast(day as integer)
# MAGIC order by year, month, day, num_visits
# MAGIC )

# COMMAND ----------

# MAGIC %md ### By separating NYC from all the other cities, the distribution looking close to normal

# COMMAND ----------

# DBTITLE 1,Distribution NYC num_visits
display(spark.sql("select * from footTrafficMedia") )

# COMMAND ----------

# MAGIC %md ### Plot features we are going to use in the model:

# COMMAND ----------

# DBTITLE 1,Use Pandas dataframe for exploratory stats analysis
city_pdf = spark.sql("select * from footTrafficMedia").toPandas()
import pandas as pd
city_pdf['date'] = pd.to_datetime(city_pdf['date'].str.strip(), format='%Y-%m-%d')
city_pdf.head(5)

# COMMAND ----------

import numpy as np

city_pdf = spark.sql("select * from (select region, cast(year as integer) year, cast(month as integer) month, cast(day as integer) day, sum(num_visits) num_visits from subway_foot_traffic  where  region = 'NY' and city = 'New York' group by region, cast(year as integer), cast(month as integer), cast(day as integer))").toPandas()

# COMMAND ----------

city_pdf['date'] = pd.to_datetime(city_pdf[["year", "month", "day"]])
city_pdf = city_pdf.sort_values('date')
# generate NYC Subway campaign media: banner impression 
# normal distr
city_pdf['banner_imp'] = np.around( np.random.randint(20000, 100000, city_pdf.shape[0]) *  np.log(city_pdf['num_visits']))

#generate NYC Subway campaign media: social media like count
# lognormal distr
city_pdf['social_media_like'] = np.around( np.random.lognormal(3, 0.25, city_pdf.shape[0]) *  city_pdf['num_visits']/1000)

# generate landing page visit
# lognormal distr + moving average
city_pdf['landing_page_visit'] = np.around( np.random.lognormal(6, 0.03, city_pdf.shape[0]) * city_pdf['num_visits']/555).rolling(window=7).mean().fillna(400)

# File location and type
file_location = "dbfs:/databricks-datasets/identifying-campaign-effectiveness/googleTrend/multiTimeline.csv"
file_type = "csv"

# CSV options
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
google_trend_df = spark.read.format(file_type) \
  .schema("Week timestamp, google_trend integer") \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location) \

# Upsample from weekly to daily
google_trend_pdf = google_trend_df.toPandas().rename(columns={'Week': 'date'})
merge=pd.merge(city_pdf,google_trend_pdf, how='left', on='date')
merge['google_trend_fill']  = merge['google_trend'].ffill()  
city_pdf = merge.fillna(82).drop('google_trend', axis=1).rename(columns={'google_trend_fill': 'google_trend'})

# COMMAND ----------

from plotly.subplots import make_subplots
import plotly.graph_objects as go

df_plot = city_pdf.copy()
fig = make_subplots( rows=5, cols=1, subplot_titles=("Store Visits", "Banner Impression", "Social Media Like", "Google Trend", "Landing Page Visit"))

fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['num_visits']), row=1, col=1, )
fig.add_trace(go.Bar(x=df_plot['date'], y=df_plot['banner_imp']), row=2, col=1)
fig.add_trace(go.Bar(x=df_plot['date'], y=df_plot['social_media_like']), row=3, col=1)
fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['google_trend']), row=4, col=1)
fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['landing_page_visit']), row=5, col=1)

fig.update_layout(height=1000, width=1200, title_text="Subway Foot Traffic Dataset")

fig.show()

# COMMAND ----------

# MAGIC %md ### Explore Features: distribution test
# MAGIC ###### As linear models like normally distributed data. The target - Store num_visits is slightly right skewed however is close to normally distributed. we will not need to transform num_visits. However, landing_page_visit is lognomal hence we perform log transformation.

# COMMAND ----------

# DBTITLE 1,Stats of the features
df_plot.describe()

# COMMAND ----------

# DBTITLE 1,Distribution Plots and Q-Q Plots
# MAGIC   %matplotlib inline
# MAGIC   import pylab as pl
# MAGIC   import seaborn as sns
# MAGIC   import matplotlib.pyplot as plt
# MAGIC   from scipy import stats
# MAGIC   sns.set(color_codes=True)
# MAGIC 
# MAGIC   fig,axes = plt.subplots(ncols=2,nrows=3)
# MAGIC   fig.set_size_inches(10, 18)
# MAGIC 
# MAGIC   sns.histplot(df_plot['num_visits'], color="g",  ax=axes[0][0], kde_kws={"color": "r", "lw": 1.5, "label": "Normal Distr fit"},)
# MAGIC   stats.probplot(df_plot['num_visits'], dist='norm', sparams=(2.5,),fit=True, plot=axes[0][1])
# MAGIC 
# MAGIC   
# MAGIC   sns.histplot(df_plot['google_trend'], color="g",  ax=axes[1][0], kde_kws={"color": "r", "lw": 1.5, "label": "Normal Distr fit"},)
# MAGIC   stats.probplot(df_plot['google_trend'], dist='norm', sparams=(2.5,),fit=True, plot=axes[1][1])
# MAGIC 
# MAGIC   # log transform
# MAGIC   sns.histplot(np.log(df_plot['landing_page_visit']), color="g",  ax=axes[2][0], kde_kws={"color": "r", "lw": 1.5, "label": "Lognormal fit"},)
# MAGIC   stats.probplot(np.log(df_plot['landing_page_visit']), dist='norm', sparams=(2.5,),fit=True, plot=axes[2][1])
# MAGIC 
# MAGIC   display(fig)

# COMMAND ----------

# MAGIC %md ## Use XGBoost to train models

# COMMAND ----------

# MAGIC %md #### Config HyperOpt, AutoML

# COMMAND ----------

from hyperopt import fmin, tpe, rand, hp, Trials, STATUS_OK
import xgboost
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
import mlflow
import mlflow.xgboost

from sklearn.model_selection import train_test_split
pdf = city_pdf.copy()
X_train, X_test, y_train, y_test = train_test_split(pdf.drop(['region',	'year',	'month','day','date', 'num_visits'], axis=1), pdf['num_visits'], test_size=0.33, random_state=55)

def train(params):
  """
  An example train method that computes the square of the input.
  This method will be passed to `hyperopt.fmin()`.
  
  :param params: hyperparameters. Its structure is consistent with how search space is defined. See below.
  :return: dict with fields 'loss' (scalar loss) and 'status' (success/failure status of run)
  """
  curr_model =  XGBRegressor(learning_rate=params[0],
                            gamma=int(params[1]),
                            max_depth=int(params[2]),
                            n_estimators=int(params[3]),
                            min_child_weight = params[4], objective='reg:squarederror')
  score = -cross_val_score(curr_model, X_train, y_train, scoring='neg_mean_squared_error').mean()
  score = np.array(score)
  
  return {'loss': score, 'status': STATUS_OK, 'model': curr_model}


# define search parameters and whether discrete or continuous
search_space = [ hp.uniform('learning_rate', 0, 1),
                 hp.uniform('gamma', 0, 5),
                 hp.randint('max_depth', 10),
                 hp.randint('n_estimators', 20),
                 hp.randint('min_child_weight', 10)
               ]
# define the search algorithm (TPE or Randomized Search)
# can choose tpe for more complex search but it requires more trials
# algo= rand.suggest
algo= tpe.suggest


from hyperopt import SparkTrials
search_parallelism = 4
spark_trials = SparkTrials(parallelism=search_parallelism)

with mlflow.start_run():
  argmin = fmin(
    fn=train,
    space=search_space,
    algo=algo,
    max_evals=8,
    trials=spark_trials)
  
mlflow.end_run()

# COMMAND ----------

# MAGIC %md #### Fit a XGBRegressor model using best set of hyperparameters

# COMMAND ----------

def fit_best_model(X, y, argmin): 
  xgb_regressor =  XGBRegressor(objective='reg:squarederror', **argmin)
  xgb_model = xgb_regressor.fit(X, y, verbose=False)
  return(xgb_model)

# COMMAND ----------

# experiment_name is the notebook_path 

from sklearn.model_selection import train_test_split
pdf = city_pdf.copy()
X_train, X_test, y_train, y_test = train_test_split(pdf.drop(['region',	'year',	'month','day','date', 'num_visits'], axis=1), pdf['num_visits'], test_size=0.33, random_state=55)

# fit model using best parameters and log the model
xgb_model = fit_best_model(X_train, y_train, argmin) 
mlflow.xgboost.log_model(xgb_model, "xgboost") # log the model here 


# Cal r2 for the best model - isn't pretty probably shouldn't show !!
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
train_pred = xgb_model.predict(X_train)
test_pred = xgb_model.predict(X_test)

print('Train r2 score: ', r2_score(train_pred, y_train))
print('Test r2 score: ', r2_score(y_test, test_pred))
train_mse = mean_squared_error(train_pred, y_train)
test_mse = mean_squared_error(test_pred, y_test)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
print('Train RMSE: %.4f' % train_rmse)
print('Test RMSE: %.4f' % test_rmse)

# COMMAND ----------

# MAGIC %md ### Use model to predict NYC Subway In Store Traffic

# COMMAND ----------

X = pdf.drop(['region',	'year',	'month','day','date', 'num_visits'], axis=1)
y = pdf["num_visits"]
pred = xgb_model.predict(X)
pdf['pred_num_visits'] = pred

import matplotlib.pyplot as plt

plt.figure(figsize=(23,5))
plt.plot(pdf.date, pdf.num_visits)
plt.plot(pdf.date, pdf.pred_num_visits, color='salmon', linestyle='--')
plt.title('Predicted Foot Traffic vs Actual')
plt.ylabel('Store visits')
plt.xlabel('Date')
plt.show()

# COMMAND ----------

# MAGIC %md ### Use SHAP to inteprate the attribution of each media channel for the in store Foot Traffic
# MAGIC - Understand how much each media input contributes to in-store foot traffic
# MAGIC - Model-based business measures: Apply SHAP to interpret the model-based outputs and look at each media channel’s effectiveness
# MAGIC - Need to install SHAP python lib

# COMMAND ----------

import shap
# load JS visualization code to notebook
shap.initjs()
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X, y=y.values)

mean_abs_shap = np.absolute(shap_values).mean(axis=0).tolist()

# display(spark.createDataFrame(sorted(list(zip(mean_abs_shap, X.columns)), reverse=True)[:8], ["Mean |SHAP|", "Feature"]))

# COMMAND ----------

# MAGIC %md #### The average contribution of Landing Page Visit, Social Media Like, Banner Impression, Google Trend (organic search)

# COMMAND ----------

shap.summary_plot(shap_values, X, plot_type="bar")

# COMMAND ----------

display(spark.createDataFrame(sorted(list(zip(mean_abs_shap, X.columns)), reverse=True)[:8], ["Mean |SHAP|", "Feature"]))

# COMMAND ----------

# MAGIC %md #### SHAP can provide the granular insight of media mix contribution to daily in store foot traffic

# COMMAND ----------

# visualize one row prediction's explanation (use matplotlib=True to avoid Javascript)
n = 105
pdf[n:n+1]

# COMMAND ----------

plot_html = shap.force_plot(explainer.expected_value, shap_values[n:n+1], feature_names=X.columns, plot_cmap='GnPR')  
displayHTML(bundle_js + plot_html.html())

# COMMAND ----------

shap_values_pdf = pd.DataFrame({'banner_imp': shap_values[:, 0], 'social_media_like': shap_values[:, 1], 'google_trend': shap_values[:, 2], 'landing_page_visit': shap_values[:, 3]})
shap_values_pdf['base_value'] = explainer.expected_value
shap_values_pdf['date'] = pdf['date']

# COMMAND ----------

# MAGIC %md ### Full Decomposition Chart of Media Contribution

# COMMAND ----------

import plotly.graph_objects as go

fig = go.Figure(data=[
    go.Bar(name='base_value', x=shap_values_pdf['date'], y=shap_values_pdf['base_value'], marker_color='lightblue'),
    go.Bar(name='banner_imp', x=shap_values_pdf['date'], y=shap_values_pdf['banner_imp']),
    go.Bar(name='social_media_like', x=shap_values_pdf['date'], y=shap_values_pdf['social_media_like']),
    go.Bar(name='landing_page_visit', x=shap_values_pdf['date'], y=shap_values_pdf['landing_page_visit']),
    go.Bar(name='google_trend', x=shap_values_pdf['date'], y=shap_values_pdf['google_trend'])
])
# Change the bar mode
fig.update_layout(barmode='stack')

fig.write_html("/dbfs/home/layla/data/tmp/plotly2.html")

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Conclusion 
# MAGIC 
# MAGIC * Based on the Contribution charts, it's easy to represent the Foot Traffic in-store visit due to each online media input. 
# MAGIC * The Social Media FB likes and Landing page visit are the most effictive channels to drive foot traffic.  
# MAGIC * The model results can be used further to perform deep dive analysis to assess the effectiveness of each campaign by understanding which campaigns or creatives work better than the other ones. It can be used to do Budget optimization. Budget allocation is shifted from low performing channels to high performing channels/genres to increase overall sales or market share.

# COMMAND ----------

# MAGIC %md
# MAGIC Copyright Databricks, Inc. [2021]. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC |Library Name|Library license | Library License URL | Library Source URL |
# MAGIC |---|---|---|---|
# MAGIC |numpy|NumPy license Copyright (c) 2005-2022, NumPy Developers| https://numpy.org/doc/stable/license.html | https://numpy.org/doc/stable/index.html
# MAGIC |plotly|MIT license|https://github.com/https://github.com/plotly/plotly.py/blob/master/LICENSE.txt|https://github.com/plotly|
# MAGIC |pandas|BSD 3-Clause License|https://github.com/pandas-dev/pandas/blob/main/LICENSE|https://github.com/pandas-dev/pandas|
# MAGIC |pylab, matplotlib|BSD compatible code, and its license is based on the PSF license|https://matplotlib.org/stable/users/project/license.html|https://matplotlib.org/stable/index.html|
# MAGIC |seaborn| BSD 3-Clause "New" or "Revised" License|https://github.com/mwaskom/seaborn/blob/master/LICENSE|https://github.com/mwaskom/seaborn|
# MAGIC |hyperopt|BSD License (BSD)|https://github.com/hyperopt/hyperopt/blob/master/LICENSE.txt|https://github.com/hyperopt/hyperopt|
# MAGIC |scipy|BSD 3-Clause "New" or "Revised" License|https://github.com/scipy/scipy/blob/main/LICENSE.txt|https://github.com/scipy/scipy|
# MAGIC |xgboost|Apache License 2.0|https://github.com/dmlc/xgboost/blob/master/LICENSE|https://github.com/dmlc/xgboost|
# MAGIC |shap|The MIT License (MIT)|https://github.com/slundberg/shap/blob/master/LICENSE|https://github.com/slundberg/shap|
# MAGIC |scikit-learn|BSD 3-Clause "New" or "Revised" License|https://github.com/scikit-learn/scikit-learn/blob/main/COPYING|https://github.com/scikit-learn/scikit-learn|
# MAGIC |mlflow|Apache-2.0 License |https://github.com/mlflow/mlflow/blob/master/LICENSE.txt|https://github.com/mlflow/mlflow|
# MAGIC |Python|Python Software Foundation (PSF) |https://github.com/python/cpython/blob/master/LICENSE|https://github.com/python/cpython|
# MAGIC |Spark|Apache-2.0 License |https://github.com/apache/spark/blob/master/LICENSE|https://github.com/apache/spark|
