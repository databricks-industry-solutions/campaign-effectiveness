# Databricks notebook source
# MAGIC %md This notebook sets up the companion cluster(s) to run the solution accelerator. It also creates the Workflow to create a Workflow DAG and illustrate the order of execution. Feel free to interactively run notebooks with the cluster or to run the Workflow to see how this solution accelerator executes. Happy exploring!
# MAGIC 
# MAGIC Note that the pipelines, workflows and clusters created in this script are not user-specific. Running this script again after modification resets them for other users too.
# MAGIC 
# MAGIC **Note**: If the job execution fails, please confirm that you have set up other environment dependencies as specified in the accelerator notebooks. Accelerators sometimes require the user to set up additional cloud infra or data access, for instance. 

# COMMAND ----------

# DBTITLE 0,Install util packages
# MAGIC %pip install git+https://github.com/databricks-industry-solutions/notebook-solution-companion git+https://github.com/databricks-academy/dbacademy-rest git+https://github.com/databricks-academy/dbacademy-gems 

# COMMAND ----------

job_json =  {
    "timeout_seconds": 7200,
    "max_concurrent_runs": 1,
    "tags": {
        "usage": "solacc_testing",
        "group": "CME"
    },
    "tasks": [
        {
            "job_cluster_key": "campaign_cluster",
            "libraries": [],
            "notebook_task": {
                "notebook_path": f"01a_Identifying Campaign Effectiveness For Forecasting Foot Traffic: ETL",
                "base_parameters": {
                    "holdout days": "90"
                }
            },
            "task_key": "campaign_01",
            "description": ""
        },
        {
            "job_cluster_key": "campaign_cluster",
            "notebook_task": {
                "notebook_path": f"01b_Campaign Effectiveness_Forecasting Foot Traffic_Machine Learning",
                "base_parameters": {
                    "holdout days": "90"
                }
            },
            "task_key": "campaign_02",
            "depends_on": [
                {
                    "task_key": "campaign_01"
                }
            ]
        }
    ],
    "job_clusters": [
        {
            "job_cluster_key": "campaign_cluster",
            "new_cluster": {
                "spark_version": "10.4.x-cpu-ml-scala2.12",
            "spark_conf": {
                "spark.databricks.delta.formatCheck.enabled": "false"
                },
                "num_workers": 2,
                "node_type_id": {"AWS": "i3.xlarge", "MSA": "Standard_D3_v2", "GCP": "n1-highmem-4"}, # different from standard API
                "custom_tags": {
                    "usage": "solacc_testing"
                },
            }
        }
    ]
}


# COMMAND ----------

from solacc.companion import NotebookSolutionCompanion

# COMMAND ----------

dbutils.widgets.dropdown("run_job", "False", ["True", "False"])
run_job = dbutils.widgets.get("run_job") == "True"
NotebookSolutionCompanion().deploy_compute(job_json, run_job=run_job)
