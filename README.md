## Identifying Campaign Effectiveness For Forecasting Foot Traffic: ETL
###### In advertising, one of the most important practices to be able to deliver to clients is information about how their advertising spend drove results -- and the more quickly we can provide the attributed information to clients, the better. To tie an offline activity to the impressions served in an advertising campaign, companies must perform attribution. Attribution can be a fairly expensive process, and running attribution against constantly updating datasets is challenging without the right technology.

###### Fortunately, Databricks makes this easy with Unified Data Analytics Platform and Delta.
#### The main steps taken in this notebook are:  

* Ingest (mock) Monthly Foot Traffic Time Series in [SafeGraph format](https://www.safegraph.com/points-of-interest-poi-data-guide) - here we've mocked data to fit the schema (Bronze)
* Convert to monthly time series data - so numeric value for number of vists per date (row = date) (Silver)
* Restrict to the NYC area for Subway Restaurant (Gold)
* Exploratory analysis of features: distribution check, variable transoformation (Gold)

More information on the SafeGraph data is below: 

#### What is SafeGraph Patterns? 
* SafeGraph's Places Patterns is a dataset of anonymized and aggregated visitor foot-traffic and visitor demographic data available for ~3.6MM points of interest (POI) in the US. 
* Here we look at historical data (Jan 2019 - Feb 2020) for a set of limited-service restaurants in-store visits

<div >
  <img src="https://databricks.com/wp-content/uploads/2020/10/mm-ref-arch-1.png" style="width:1100px;height:550px;">
</div>

To run this accelerator, clone this repo into a Databricks workspace. Attach the RUNME notebook to any cluster running a DBR 11.0 or later runtime, and execute the notebook via Run-All. A multi-step-job describing the accelerator pipeline will be created, and the link will be provided. Execute the multi-step-job to see how the pipeline runs. 

The job configuration is written in the RUNME notebook in json format. The cost associated with running the accelerator is the user's responsibility.
