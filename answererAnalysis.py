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

#GRAB DATA
#convert table to sparks df 
df = spark.table("default.table_addintagAnswers")
#convert spark df to pandas df 
pdf = df.select("*").toPandas() 

# COMMAND ----------

#NOTE: important vocab -- answer: an answer from the community ; answerer: a user that answers questions 
#FILTER DATA 
from datetime import datetime, timedelta, date
import pandas as pd

currentDate = date.today()  #date the the "tagAnswerData" script was last run (should be run before this script)
microAnswerer = ["Alexander Jerabek", "Juan Balmori", "Kelbow (MSFT)", "Kim Brandl", "Rick Kirkham", "Wamwitha Love", "Nitesh"]

currentfilterdf = pdf[pdf['current_date'] == currentDate] #you want to filter by current date, because whenever the "tagAnswerData" script is run, new data for that day is 
#there is nothing stored for 30th
currentfilterdf = currentfilterdf[currentfilterdf['queried_tag'] == "office-js"] #get all questions for office js 

displayNames = list(currentfilterdf["owner.display_name"])

totalAnswerer = [] #list of all answerers
nonMicroAnswerer = [] #list of all community answerers
answers = len(displayNames) #count of all answers (number of answers is number of display names)
nonMicroAnswer = 0 #count of answers written my nonMicro employees 

for displayName in displayNames: 
  if (displayName not in totalAnswerer): #creating list of userids in community 
      totalAnswerer.append(displayName)
  if ((displayName not in microAnswerer) & ("MSFT" not in displayName) & ("Microsoft" not in displayName)): #number of answers not from microsoft employee
      nonMicroAnswer += 1
  if ((displayName not in microAnswerer) & (displayName not in nonMicroAnswerer) & ("MSFT" not in displayName) & ("Microsoft" not in displayName)): #number of non-micro answerers
      nonMicroAnswerer.append(displayName)

percentOfCommunityAnswers = (nonMicroAnswer/answers) * 100 #percent of answers from community 

nonMicroAnswererSize = len(nonMicroAnswerer) #creating variable to reduce len() method call
percentOfCommunityAnswerers = (nonMicroAnswererSize/len(totalAnswerer)) * 100 #percent of users from community 

df = pd.DataFrame([(currentDate, percentOfCommunityAnswerers, percentOfCommunityAnswers, nonMicroAnswererSize)])
df.columns = ['date', 'percentOfCommunityAnswerers', 'percentOfCommunityAnswers', 'raw_countOfCommunityAnswerers']

print(df)

# COMMAND ----------

df["date"] = pd.to_datetime(df["date"])
df["raw_countOfCommunityAnswerers"] = pd.to_numeric(df["raw_countOfCommunityAnswerers"])
df["percentOfCommunityAnswerers"] = pd.to_numeric(df["percentOfCommunityAnswerers"])
df["percentOfCommunityAnswers"] = pd.to_numeric(df["percentOfCommunityAnswers"])

print (df)
spark_df = pandas_to_spark(df)
spark_df.write.mode("append").saveAsTable("officejsdata")
