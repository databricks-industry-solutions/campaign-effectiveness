# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

spark.conf.set("spark.databricks.io.cache.enabled", "true")

# COMMAND ----------

# MAGIC %sh 
# MAGIC cd /tmp/
# MAGIC wget http://databricks.com/notebooks/safegraph_patterns_simulated__1_-91d51.csv
# MAGIC cp safegraph_patterns_simulated__1_-91d51.csv /dbfs/tmp/altdata_poi/foot_traffic.csv

# COMMAND ----------

# MAGIC %fs mkdirs /home/layla/data/tmp/foot_traffic_data/

# COMMAND ----------

# MAGIC %sh 
# MAGIC wget -P /dbfs/home/layla/data/tmp/foot_traffic_data/ https://github.com/databricks/tech-talks/blob/master/datasets/county-estimates.csv

# COMMAND ----------

dates=spark.sql("""select '201901' year_month, cast(cast('2019-01-01' as timestamp) as long) date_range_start, cast(cast('2019-01-31' as timestamp) as long) date_range_end
union all 
select '201902' year_month, cast(cast('2019-02-01' as timestamp) as long) date_range_start, cast(cast('2019-02-28' as timestamp) as long) date_range_end
union all
select '201903' year_month, cast(cast('2019-03-01' as timestamp) as long) date_range_start, cast(cast('2019-03-31' as timestamp) as long) date_range_end
union all
select '201904' year_month, cast(cast('2019-04-01' as timestamp) as long) date_range_start, cast(cast('2019-04-30' as timestamp) as long) date_range_end
union all
select '201905' year_month, cast(cast('2019-05-01' as timestamp) as long) date_range_start, cast(cast('2019-05-31' as timestamp) as long) date_range_end
union all
select '201906' year_month, cast(cast('2019-06-01' as timestamp) as long) date_range_start, cast(cast('2019-06-30' as timestamp) as long) date_range_end
union all
select '201907' year_month, cast(cast('2019-07-01' as timestamp) as long) date_range_start, cast(cast('2019-07-31' as timestamp) as long) date_range_end
union all
select '201908' year_month, cast(cast('2019-08-01' as timestamp) as long) date_range_start, cast(cast('2019-08-31' as timestamp) as long) date_range_end
union all
select '201909' year_month, cast(cast('2019-09-01' as timestamp) as long) date_range_start, cast(cast('2019-09-30' as timestamp) as long) date_range_end
union all
select '201910' year_month, cast(cast('2019-10-01' as timestamp) as long) date_range_start, cast(cast('2019-10-31' as timestamp) as long) date_range_end
union all
select '201911' year_month, cast(cast('2019-11-01' as timestamp) as long) date_range_start, cast(cast('2019-11-30' as timestamp) as long) date_range_end
union all
select '201912' year_month, cast(cast('2019-12-01' as timestamp) as long) date_range_start, cast(cast('2019-12-31' as timestamp) as long) date_range_end
union all
select '202001' year_month, cast(cast('2020-01-01' as timestamp) as long) date_range_start, cast(cast('2020-01-31' as timestamp) as long) date_range_end
union all
select '202002' year_month, cast(cast('2020-02-01' as timestamp) as long) date_range_start, cast(cast('2020-02-29' as timestamp) as long) date_range_end""")

# COMMAND ----------

import re
db_prefix = "campaign_effectiveness"
current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
current_user_no_at = current_user[:current_user.rfind('@')]
current_user_no_at = re.sub(r'\W+', '_', current_user_no_at)

dbName = db_prefix+"_"+current_user_no_at
cloud_storage_path = f"/Users/{current_user}/solution_accelerator/{db_prefix}/"
reset_all = dbutils.widgets.get("reset_all_data") == "true"

if reset_all:
  spark.sql(f"DROP DATABASE IF EXISTS {dbName} CASCADE")
  dbutils.fs.rm(cloud_storage_path, True)
  dbutils.fs.mkdirs(cloud_storage_path)

spark.sql(f"""create database if not exists {dbName} LOCATION '{cloud_storage_path}/tables' """)
spark.sql(f"""USE {dbName}""")

print("using cloud_storage_path {}".format(cloud_storage_path))
print("using database {arg1} with location at {arg2}{arg3}".format(arg1= dbName,arg2= cloud_storage_path, arg3='tables/'))
