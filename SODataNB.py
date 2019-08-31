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

from pandas.io.json import json_normalize
import requests
from datetime import datetime
import pandas as pd 

tags = ["officedev", "openoffice-writer" , "excel-addins" , "word-addins", "outlook-addin", "visual-studio-addins", "outlook-web-addins", "office-addins", "sharepoint-addin", "powerpoint-addins", "visual-studio-addins", "word-web-addins", "office-js"]
page = 1
hasMore = True
filterString = "!LGnKD9j.jC*_Lr0VPqEi)*" #customized filter for retrieving specific data fields 
APIKey = "7mvb74ZDBuXRUL05SEDWQQ(("
finalDF = pd.DataFrame()

for tag in tags:
  page = 1
  hasMore = True
  while (hasMore):
    url = "https://api.stackexchange.com/2.2/search?page=" + str(page) + "&pagesize=100&order=desc&sort=activity&tagged=" + tag +"&site=stackoverflow&filter=" + filterString + "&key=" + APIKey
    request = requests.get(url).json()
    items = request.get('items') 
    
    #convert JSON to df 
    tempdf = json_normalize(items) #note: dates are in unix epoch time stored as bigints 
    
    #add rows to df 
    tempdf['queried_tag'] = tag
    tempdf['current_date'] = datetime.now() #UTC 
    
    #edit type of current rows 
    tempdf["creation_date"] = pd.to_datetime(tempdf["creation_date"], unit='s') #dates are in UTC 
    tempdf["last_edit_date"] = pd.to_datetime(tempdf["last_edit_date"], unit='s')
    tempdf["last_activity_date"] = pd.to_datetime(tempdf["last_activity_date"], unit='s')
    tempdf['current_date'] = datetime.now()
    
    finalDF = finalDF.append(tempdf)
     
    page += 1
    hasMore = request.get('has_more') 
    
spark_df = pandas_to_spark(finalDF)
spark_df.write.mode("append").saveAsTable("default.table_stackOverflowdata")
