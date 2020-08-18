from flask import Flask, render_template
import numpy as np
import pandas as pd
import os
import joblib as jb
import extract_data as ed
import processing_data as proc
from flask import jsonify
import json
from flask import request, redirect, url_for

app = Flask(__name__,template_folder='template')

model_lgbm = jb.load('lgb.pkl.z')
title_vec = jb.load('title_vec.pkl.z')
sinopse_vec = jb.load('sinopse_vec.pkl.z')


urls = ['https://www.imdb.com/search/title/?genres=action&start={itens}&explore=title_type,genres&ref_=adv_nxt',
        'https://www.imdb.com/search/title/?genres=horror&start={itens}&explore=title_type,genres&ref_=adv_nxt']

def get_data(url=urls,itens=200):
    print(url)
    ed.get_data_from_imdb(url,itens)    
    df = pd.read_csv('new_movies.csv')    
    df = proc.preprocessing()
    df.to_csv('movies_ok.csv',index=False)
    
def get_prediction():    
    df = pd.read_csv('movies_ok.csv')
    v_title = title_vec.transform(df['title'])
    v_sinopse = sinopse_vec.transform(df['sinopse'])
    df_title = pd.DataFrame(data=v_title.toarray(),columns=title_vec.get_feature_names())
    df_sinopse = pd.DataFrame(data=v_sinopse.toarray(),columns=sinopse_vec.get_feature_names())    
    df=pd.concat([df,df_title,df_sinopse],axis=1)
    x = df.drop(columns=['title','sinopse','link']).values    
    y_pred = model_lgbm.predict_proba(x)
    return df,y_pred
    
@app.route('/movies/get_data')
def start_extract_data():
    if not os.path.exists('new_movies.csv'):
        get_data(urls)
        return jsonify(201)
    else:
        return jsonify({'status':200, 'message':'Já existe o arquivo de novos filmes'})
        
@app.route('/',methods=['GET','POST'])
def prediction(redirecionar=False):      
    df,y_pred = get_prediction()        
    df['probabilidade'] = y_pred[:,1]
    df['probabilidade'] = df.apply(lambda row: round(row['probabilidade']*100,2),axis=1)
    df.sort_values(by='probabilidade',ascending=False,inplace=True)
    movies = json.loads(df[['title','probabilidade','link']].head(15).to_json(orient='records'))
    if not redirecionar:
        return render_template('predict.html',movies=movies)
    else:
        return jsonify({'status':200,'movies':movies})        

@app.route('/movies/<string:link>',methods=['GET','POST'])
def pred_with_link(link):    
    linkw = request.get_json()
    links=[]
    links.append(linkw['link'])
    get_data(links,300)
    return prediction(True)
    
    
    
if __name__=='__main__':
   app.run(debug=True,use_reloader=False)

    

