import praw
import datetime
import pandas as pd
import torch
import requests
import json
import matplotlib.pyplot as plt
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np


def sym(keyword):
    a={'airtel':"BHARTARTL","gamestop":"GME","adani stock":"ADANIENT"}
    if keyword.lower() in a.keys():
        return a[keyword.lower()]
    # Set the API endpoint URL
    endpoint = "https://www.alphavantage.co/query"
    api="PWCXEB3R143MU162"
    # # Set the API parameters
    params = {
        "function": "SYMBOL_SEARCH",
        "keywords": keyword,
        "apikey": api
    }
    response = requests.get(endpoint, params=params)
    data = response.json()
    # formatted_data = json.dumps(data, indent=4)
    symbols = [result["1. symbol"] for result in data["bestMatches"]]
    return symbols[0]


# def getcomments(id):
#     submission = reddit.submission(id=id)

#     submission.comments.replace_more(limit=1)  # Retrieve all comments, including nested ones
#     comments=[]
#     for comment in submission.comments.list():
#         # print(comment.body)
#         comments.append(comment.body)
#     print(len(comments))


def scrape_reddit(search_query):
    secret= "xV3O4nwRdpRhkp-q3wQEqAPgr3c2sQ"
    appid = "9zVDXji3QYmrL6_ZhmenNw"
    reddit = praw.Reddit(client_id=appid,client_secret=secret,user_agent='scraper_api')
    print("Authentication is done.")
    sortsub = "hot"

    def get_data_dict(search_term, sortsub):
        data_dict = {
        "subject": list(),
        "Title" : list(),
        "score": list(),
        # "num_comments" : list(),
        "id" : list(),
        "subreddit" : list(),
        "time_stamp" : list()
        }
        print(f"This is for the searchterm {search_term}")

        for index, submission in enumerate(reddit.subreddit('all').search(search_term.lower(), sort=sortsub,limit=200)):
            
            data_dict["subject"].append(search_term.lower())
            data_dict["Title"].append(submission.title)
            data_dict["score"].append(submission.score)
            # data_dict["num_comments"].append(submission.num_comments)
            data_dict["id"].append(submission.id)
            data_dict["subreddit"].append(submission.subreddit)
            data_dict["time_stamp"].append(datetime.datetime.utcfromtimestamp(submission.created).strftime('%m-%d-%Y'))
            if index%50==0:
                print("Finished searching through ",index," posts")
        return data_dict
    
    def get_dataframe(search_query):
        df= pd.DataFrame(get_data_dict(search_term = search_query, sortsub = sortsub))
        return df
        

    df = get_dataframe(search_query)
    return df


def perform_sentiment_analysis(df):
    # sentiment_pipeline = pipeline("sentiment-analysis")
    model_id = "textattack/distilbert-base-uncased-SST-2"
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    def senti_analyze(text):
        a=sentiment_pipeline(text)
        return a

    df["sentiment"] = df["Title"].apply(senti_analyze)

    # FOR HUGGING FACE ANALYSIS
    idfk = df['sentiment'].to_numpy()
    label=[]
    score=[]
    for index,x in enumerate(idfk):
        label.append(x[0]['label'])
        score.append(x[0]['score'])
        if index%50==0:
            print("Finished doing sentiment analysis of ",index," post titles")
    df['label']=label
    df['score']=score
    df_new = df.iloc[:, [1,2,-3,-1]]

    df_new=df_new.drop_duplicates()
    new_row = ["Title","Score","Date","Label"]
    df_new.loc[0] = new_row

    df_new.sort_values(by='time_stamp', inplace = True)
    dates=[]
    negs=[]
    pos=[]
    for index, row in df_new.iterrows():
        if row['time_stamp'] not in dates:
            dates.append(row['time_stamp'])
            pos.append(0)
            negs.append(0)
        if row["label"]== "LABEL_0":
            negs[-1]=negs[-1]+1
        elif row["label"]== "LABEL_1":
            pos[-1]=pos[-1]+1

    dfsent = pd.DataFrame()
    dfsent['Date']=dates
    dfsent['Pos']=pos
    dfsent['Negs']=negs
    dfsent = dfsent.iloc[:-1 , :]
    return df_new, dfsent



def stockdata(symbol="TSLA"):
    api="PWCXEB3R143MU162"
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={api}'

    response = requests.get(url)
    data = response.json()

    # Extract the time series data from the response
    # time_series = data['Daily Time Series']
    # print(data)

    # Extract the time series data from the response
    time_series = data['Time Series (Daily)']

    # Convert the time series data into a pandas DataFrame
    dfx = pd.DataFrame.from_dict(time_series, orient='index')
    dfx = dfx[['1. open']]
    # Sort the DataFrame by date in ascending order
    dfx = dfx.sort_index(ascending=True)
    # Filter the DataFrame to include the last 2 years of data
    dfx = dfx.tail(2 * 365)
        
    dates = dfx.index
    dfx['Date']=dates
    # print(dfx.columns)
    return dfx


# My code
# df =scrape_reddit("Tesla")
# df_new,dfsent = perform_sentiment_analysis(df)
# print(sym("Alphabet"))
# df= stockdata("TSLA")
# print(df)