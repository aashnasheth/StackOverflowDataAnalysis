# Databricks notebook source
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

from pyspark.sql.types import *
from datetime import datetime, timedelta, date
import pandas as pd 
import numpy as np

tags = ["excel-addins", "openoffice-writer" , "officedev" , "word-addins", "outlook-addin", "visual-studio-addins", "outlook-web-addins", "office-addins", "sharepoint-addin", "powerpoint-addins", "visual-studio-addins", "word-web-addins", "office-js"]

#convert table to sparks df 
df = spark.table("default.table_stackoverflowdata")

#convert spark df to pandas df 
pdf = df.select("*").toPandas() 

# COMMAND ----------

from datetime import datetime, timedelta, date #dont need to track total number of questions by date - more like week or year 
                                               # schedule this every monday

currentDate = date.today() #date today
previousDate = currentDate - timedelta(days=7) #minus 7 

currentfilterdf = pdf[pdf['current_date'] == currentDate]
previousfilterdf = pdf[pdf['current_date'] == previousDate]

finalDf = pd.DataFrame()
for tag in tags:
  curr = currentfilterdf[currentfilterdf['queried_tag'] == tag].shape[0]
  
  prev = previousfilterdf[previousfilterdf['queried_tag'] == tag].shape[0]

  if (prev != 0): #if NOT empty do the following
    delta = abs(curr-prev)
    deltaTableDF = pd.DataFrame([(tag, previousDate, delta, curr, currentDate)])
    deltaTableDF.columns = ['tag', 'weekOf', 'questionDelta', 'totalTillDate', 'current_date']
    finalDf = finalDf.append(deltaTableDF, ignore_index=True)

print(finalDf)

# COMMAND ----------

finalDf["weekOf"] = pd.to_datetime(finalDf["weekOf"])
finalDf["current_date"] = pd.to_datetime(finalDf["current_date"])

spark_df = pandas_to_spark(finalDf)
print (spark_df)
spark_df.write.mode("append").saveAsTable("default.totalNOQdelta") #total number of questions delta 
