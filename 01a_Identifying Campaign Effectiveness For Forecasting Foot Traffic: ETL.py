# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/campaign-effectiveness. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/sales-forecasting

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Identifying Campaign Effectiveness For Forecasting Foot Traffic: ETL
# MAGIC ###### In advertising, one of the most important practices to be able to deliver to clients is information about how their advertising spend drove results -- and the more quickly we can provide the attributed information to clients, the better. To tie an offline activity to the impressions served in an advertising campaign, companies must perform attribution. Attribution can be a fairly expensive process, and running attribution against constantly updating datasets is challenging without the right technology.
# MAGIC 
# MAGIC ###### Fortunately, Databricks makes this easy with Unified Data Analytics Platform and Delta.
# MAGIC #### The main steps taken in this notebook are:  
# MAGIC 
# MAGIC * Ingest (mock) Monthly Foot Traffic Time Series in [SafeGraph format](https://www.safegraph.com/points-of-interest-poi-data-guide) - here we've mocked data to fit the schema (Bronze)
# MAGIC * Convert to monthly time series data - so numeric value for number of visits per date (row = date) (Silver)
# MAGIC * Restrict to the NYC area for Subway Restaurant (Gold)
# MAGIC * Exploratory analysis of features: distribution check, variable transformation (Gold)
# MAGIC 
# MAGIC More information on the SafeGraph data is below: 
# MAGIC 
# MAGIC #### What is SafeGraph Patterns? 
# MAGIC * SafeGraph's Places Patterns is a dataset of anonymized and aggregated visitor foot-traffic and visitor demographic data available for ~3.6MM points of interest (POI) in the US. 
# MAGIC * Here we look at historical data (Jan 2019 - Feb 2020) for a set of limited-service restaurants in-store visits
# MAGIC 
# MAGIC <div >
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2020/10/mm-ref-arch-1.png" style="width:1100px;height:550px;">
# MAGIC </div>

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $reset_all_data=true

# COMMAND ----------

# MAGIC %md ##### Notes:
# MAGIC * `dbName` variable contains the database name; `cloud_storage_path` is variable for the storage directory

# COMMAND ----------

raw_sim_ft = spark.read.format("csv").option("header", "true").option("sep", ",").load("dbfs:/databricks-datasets/identifying-campaign-effectiveness/subway_foot_traffic/foot_traffic.csv")

raw_sim_ft.createOrReplaceTempView("safegraph_sim_foot_traffic")

raw_sim_ft.repartition(1).write.mode("overwrite").format("csv").save(cloud_storage_path ,header = 'true')

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS subway_foot_traffic;
# MAGIC 
# MAGIC CREATE TABLE subway_foot_traffic  
# MAGIC USING parquet
# MAGIC LOCATION 's3://db-gtm-industry-solutions/data/cme/visit_attribution/subway_foot_traffic/'

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Examine raw Foot Traffic Data ingestion (Bronze)

# COMMAND ----------

# DBTITLE 1,Read CSV Foot Traffic Data
raw_sim_ft = spark.read.format("csv").option("header", "true").option("sep", ",").load("dbfs:/databricks-datasets/identifying-campaign-effectiveness/subway_foot_traffic/foot_traffic.csv")

raw_sim_ft.createOrReplaceTempView("safegraph_sim_foot_traffic")

# COMMAND ----------

# MAGIC %sql select * from safegraph_sim_foot_traffic

# COMMAND ----------

# DBTITLE 1,Write to Bronze Delta Table
bronze_path= f"{cloud_storage_path}tables/footTrafficBronze/" 
# dbutils.fs.mkdirs(bronze_path)
raw_sim_ft.write.format('delta').mode('overwrite').save(bronze_path)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Understand in store visits in different MSA area (Silver)
# MAGIC 
# MAGIC ###### Feature engineering: Note `visit_by_date` is an Array; we need to explode the data into separate rows

# COMMAND ----------

# DBTITLE 1,Add in MSA and Month/Year
safegraph_patterns = spark.sql("""
select x.*, INT(YEAR(FROM_UNIXTIME(date_range_start))) as year, 
                                        INT(MONTH(FROM_UNIXTIME(date_range_start))) as month, 
                                       case when region in ('NY', 'PA', 'NJ') then 'NYC MSA' else 'US' end msa, location_name
from safegraph_sim_foot_traffic x""")

# COMMAND ----------

# DBTITLE 1,Expand data to Show Visits By Day in Separate Rows
# Function to extract ARRAY or JSON columns for deeper analysis
from pyspark.sql.functions import udf, explode
from pyspark.sql.types import *
from pyspark.sql.functions import *
import json

def parser(element):
  return json.loads(element)

def parser_maptype(element):
  return json.loads(element, MapType(StringType(), IntegerType()))

jsonudf = udf(parser, MapType(StringType(), IntegerType()))

convert_array_to_dict_udf = udf(lambda arr: {idx: x for idx, x in enumerate(json.loads(arr))}, MapType(StringType(), IntegerType()))

def explode_json_column_with_labels(df_parsed, column_to_explode, key_col="key", value_col="value"):
  df_exploded = df_parsed.select("safegraph_place_id", "location_name", "msa", "date_range_start", "year", "month", "date_range_end", explode(column_to_explode)).selectExpr("safegraph_place_id", "date_range_end", "location_name","msa", "date_range_start", "year", "month", "key as {0}".format(key_col), "value as {0}".format(value_col))
  return(df_exploded)

def explode_safegraph_json_column(df, column_to_explode, key_col="key", value_col="value"):
  df_parsed = df.withColumn("parsed_"+column_to_explode, jsonudf(column_to_explode))
  df_exploded = explode_json_column_with_labels(df_parsed, "parsed_"+column_to_explode, key_col=key_col, value_col=value_col)
  return(df_exploded)

def explode_safegraph_array_colum(df, column_to_explode, key_col="index", value_col="value"):
  df_prepped = df.select("safegraph_place_id", "location_name", "msa", "date_range_start", "year", "month", "date_range_end", column_to_explode).withColumn(column_to_explode+"_dict", convert_array_to_dict_udf(column_to_explode))
  df_exploded = explode_json_column_with_labels(df_prepped, column_to_explode=column_to_explode+"_dict", key_col=key_col, value_col=value_col)
  return(df_exploded)

def explode_safegraph_visits_by_day_column(df, column_to_explode, key_col="index", value_col="value"):
  df_exploded = explode_safegraph_array_colum(df, column_to_explode, key_col=key_col, value_col=value_col)
  df_exploded = df_exploded.withColumn(key_col, col(key_col) + 1) # 1-indexed instead of 0-indexed
  return(df_exploded)


visits_by_day = explode_safegraph_visits_by_day_column(safegraph_patterns, column_to_explode="visits_by_day", key_col="day", value_col="num_visits")
print(visits_by_day.count())
visits_by_day.createOrReplaceTempView('visits_by_day')

# COMMAND ----------

# MAGIC %sql select * from visits_by_day 

# COMMAND ----------

# DBTITLE 1,Write to Silver Delta Table
silver_path= f"{cloud_storage_path}tables/footTrafficSilver" 
visits_by_day.write.format('delta').mode('overwrite').save(silver_path)
spark.sql(f"""CREATE TABLE IF NOT EXISTS footTrafficSilver USING DELTA LOCATION '{silver_path}' """)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Enrich the data with Subway Media data (Gold - Analytics Ready)
# MAGIC 
# MAGIC 
# MAGIC ##### Bring in various channels of media data: `banner impression`, `social media FB likes`, `web landing page visit`, `google trend` 

# COMMAND ----------

# DBTITLE 1,Subset the data to one particular Restaurant
# MAGIC %sql 
# MAGIC create
# MAGIC or replace temp view ft_raw as
# MAGIC select
# MAGIC   *
# MAGIC from
# MAGIC   visits_by_day;
# MAGIC   
# MAGIC create table if not exists subway_foot_traffic_enrich as
# MAGIC select
# MAGIC   location_name,
# MAGIC   msa,
# MAGIC   year,
# MAGIC   month,
# MAGIC   day,
# MAGIC   num_visits
# MAGIC from
# MAGIC   ft_raw
# MAGIC where
# MAGIC   location_name = 'Subway';
# MAGIC select
# MAGIC   *
# MAGIC from
# MAGIC   subway_foot_traffic;

# COMMAND ----------

# DBTITLE 1,Load NYC Subway campaign media data; merge to foot traffic dataset
import numpy as np

city_pdf = spark.sql("select * from (select region, cast(year as integer) year, cast(month as integer) month, cast(day as integer) day, sum(num_visits) num_visits from subway_foot_traffic where  region = 'NY' and city = 'New York' group by region, cast(year as integer), cast(month as integer), cast(day as integer))").toPandas()

import pandas as pd
import numpy as np
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

# COMMAND ----------

# MAGIC %md ##### Download [Google Trend data](https://trends.google.com/trends/explore?date=2019-01-01%202020-02-29&geo=US-NY-501&q=%2Fm%2F0f7q4) as index of organic search: can stream in by calling the Google Trend API using [`pytrends`](https://pypi.org/project/pytrends/)

# COMMAND ----------

# File location and type
file_location = "dbfs:/databricks-datasets/identifying-campaign-effectiveness/googleTrend/multiTimeline.csv"
file_type = "csv"

# CSV options
first_row_is_header = "true"
delimiter = ","

from pyspark.sql.functions import expr
# The applied options are for CSV files. For other file types, these will be ignored.
google_trend_df = spark.read.format(file_type) \
  .schema("Week timestamp, google_trend integer") \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(google_trend_df)

# Upsample from weekly to daily
google_trend_pdf = google_trend_df.toPandas().rename(columns={'Week': 'date'})
# google_trend_pdf['date'] = pd.to_datetime(google_trend_pdf['date'].str.strip(), format='%Y-%m-%d') 
merge=pd.merge(city_pdf,google_trend_pdf, how='left', on='date')
merge['google_trend_fill']  = merge['google_trend'].ffill()  
city_pdf = merge.fillna(82).drop('google_trend', axis=1).rename(columns={'google_trend_fill': 'google_trend'})

# COMMAND ----------

sdf = spark.createDataFrame(city_pdf)
sdf = sdf.withColumn("date", date_format(col("date"), "yyyy-MM-dd"))  #change to clean date format
display(sdf)

# COMMAND ----------

# DBTITLE 1,Daily TS chart of NYC Subway Foot Traffic from Jan-2019 to Feb-2020
display(sdf)

# COMMAND ----------

# DBTITLE 1,Write to Delta and create Gold Table
gold_path= f"{cloud_storage_path}tables/footTrafficGold"
sdf.write.format('delta').mode('overwrite').save(gold_path)
spark.sql(f"""CREATE TABLE IF NOT EXISTS footTrafficGold USING DELTA LOCATION '{gold_path}' """)

# COMMAND ----------

# MAGIC %sql select * from footTrafficGold

# COMMAND ----------

# MAGIC %md
# MAGIC Copyright Databricks, Inc. [2021]. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC |Library Name|Library license | Library License URL | Library Source URL |
# MAGIC |---|---|---|---|
# MAGIC |numpy|NumPy license Copyright (c) 2005-2022, NumPy Developers| https://numpy.org/doc/stable/license.html | https://numpy.org/doc/stable/index.html
# MAGIC |Python|Python Software Foundation (PSF) |https://github.com/python/cpython/blob/master/LICENSE|https://github.com/python/cpython|
# MAGIC |Spark|Apache-2.0 License |https://github.com/apache/spark/blob/master/LICENSE|https://github.com/apache/spark|
