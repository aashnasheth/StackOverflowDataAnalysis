# Databricks notebook source
df = spark.table("default.table_addintagAnswers")
#convert spark df to pandas df 
pdf = df.select("*").toPandas() 

# COMMAND ----------

from pandas.io.json import json_normalize
import requests
import pandas as pd
from datetime import datetime, timedelta, date
import time

currentDate = date.today()

currentfilterdf = pdf[pdf['current_date'] == currentDate] #you want to filter by current date, because whenever the "tagAnswerData" script is run, new data for that day is 
#there is nothing stored for 30th
currentfilterdf = currentfilterdf[(currentfilterdf['queried_tag'] == "outlook-addin") | (currentfilterdf['queried_tag'] == "outlook-web-addins") | (currentfilterdf['queried_tag'] == "excel-addins") | (currentfilterdf['queried_tag'] == "word-addins")] #can filter for certain tags, or simply remove the filter all together by commenting this line out

userIDs = list(currentfilterdf["owner.user_id"]) #NOTE: not scalable for many users bc only 10000 requests a day.. unless you just do this once... this makes more sense
APIKey = "7mvb74ZDBuXRUL05SEDWQQ(("
tempdf = pd.DataFrame()

checkedIDs = []
commonTagNamesDict = {}
for uid in userIDs:                                                                                                             
  if uid not in checkedIDs: 
    checkedIDs.append(uid)
    uid = format(uid, '.0f')                          
    url = "https://api.stackexchange.com/2.2/users/"+ str(uid) +"/top-answer-tags?pagesize=3&site=stackoverflow&key=" + APIKey
    time.sleep(0.04) #pause is needed otherwise api will cut off request limit (2000 uid means around 4 mins of waiting)
    request = requests.get(url).json()
    items = request.get('items') 
    if (items != None):
      for i in items:
        tagName =  i.get("tag_name")
        if tagName not in commonTagNamesDict.keys(): 
          commonTagNamesDict.update( {tagName : 0} )
        commonTagNamesDict[tagName] += 1 

print (commonTagNamesDict)

# COMMAND ----------

df = pd.DataFrame(list(commonTagNamesDict.items()), columns=['tag', 'count'])
df["count"] = pd.to_numeric(df["count"])
df = df.sort_values(by=['count'], ascending=False)
print (df)
