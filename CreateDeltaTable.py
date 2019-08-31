# Databricks notebook source
from pyspark.sql.types import *
from datetime import datetime, timedelta, date
import pandas as pd 
import numpy as np

tags = ["excel-addins", "openoffice-writer" , "officedev" , "word-addins", "outlook-addin", "visual-studio-addins", "outlook-web-addins", "office-addins", "sharepoint-addin", "powerpoint-addins", "word-web-addins", "office-js"]

#convert table to sparks df 
df = spark.table("default.table_stackoverflowdata")

#convert spark df to pandas df 
pdf = df.select("*").toPandas() 

# COMMAND ----------

from datetime import datetime, timedelta, date

currentDate = date.today() 
previousDate = currentDate - timedelta(days=1)


currentfilterdf = pdf[pdf['current_date'] == currentDate]
currentfilterdf = currentfilterdf.groupby('queried_tag', as_index=False).agg({"view_count": "sum"})

previousfilterdf = pdf[pdf['current_date'] == previousDate]
previousfilterdf = previousfilterdf.groupby('queried_tag', as_index=False).agg({"view_count": "sum"})

finalDf = pd.DataFrame()
for tag in tags:
  curr = currentfilterdf[currentfilterdf['queried_tag'] == tag]
  prev = previousfilterdf[previousfilterdf['queried_tag'] == tag]
  
  if (prev.empty == False): #if NOT empty do the following
    delta = abs(curr['view_count'].iloc[0] - prev['view_count'].iloc[0])
    deltaTableDF = pd.DataFrame([(tag, previousDate, delta)])
    deltaTableDF.columns = ['tag', 'date', 'delta']
    finalDf = finalDf.append(deltaTableDF, ignore_index=True)

# COMMAND ----------

#module from Gonzalo on StackOverflow -- algorithm fixes error when converting from pandasdf to sparksdf because schema can't be read properly

from pyspark.sql.types import *

# Auxiliar functions
def equivalent_type(f):
    if f == 'datetime64[ns]': return DateType()
    elif f == 'int64': return LongType()
    elif f == 'int32': return IntegerType()
    elif f == 'float64': return FloatType()
    else: return StringType()

def define_structure(string, format_type):
    try: typo = equivalent_type(format_type)
    except: typo = StringType()
    return StructField(string, typo)

# Given pandas dataframe, it will return a spark's dataframe.
def pandas_to_spark(pandas_df):
    columns = list(pandas_df.columns)
    types = list(pandas_df.dtypes)
    struct_list = []
    i = 0
    for column, typo in zip(columns, types): 
      struct_list.append(define_structure(column, typo))
    p_schema = StructType(struct_list)
    return sqlContext.createDataFrame(pandas_df, p_schema)

# COMMAND ----------

finalDf["date"] = pd.to_datetime(finalDf["date"])
#finalDf["delta"] = pd.to_numeric(finalDf["delta"])

spark_df = pandas_to_spark(finalDf)
spark_df.write.mode("append").saveAsTable("default.table_SODeltaData")
