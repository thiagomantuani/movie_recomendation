import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import nltk
import unidecode
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
import joblib as jb


nltk.download('popular')
stop = nltk.corpus.stopwords.words('english')

df = pd.read_csv('movies_with_label.csv')

df.drop(columns=['genre','link','sinopse_br','gen_pred'],inplace=True)

df['runtime'].fillna(0,inplace=True)
#df = pd.get_dummies(df,columns=['gen_pred'],drop_first=True)

cols = ['rating','runtime','votes','year']
for c in cols:
    df[c] = StandardScaler().fit_transform(df[c].values.reshape(-1,1))

df['title'] = df.apply(lambda row: unidecode.unidecode(row['title']),axis=1)
df['sinopse'] = df.apply(lambda row: unidecode.unidecode(row['sinopse']),axis=1)
    
vectorizer_t = TfidfVectorizer(ngram_range=(1,1),stop_words=stop, min_df=2)    
vectorizer_s = TfidfVectorizer(ngram_range=(1,1),stop_words=stop, min_df=2)    

tit_vectorzer = vectorizer_t.fit_transform(df['title'])
df_title = pd.DataFrame(data=tit_vectorzer.toarray(),columns=vectorizer_t.get_feature_names())

sinopse_vect = vectorizer_s.fit_transform(df.sinopse)
df_sinopse = pd.DataFrame(data=sinopse_vect.toarray(),columns=vectorizer_s.get_feature_names())

df = pd.concat([df_title,df_sinopse,df],axis=1)

x = df.drop(columns=['target','title','sinopse']).values
y = df.target.values


x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=20)


md = LGBMClassifier(n_estimators=1000)
md.fit(x_train,y_train)
y_pred = md.predict_proba(x_test)

print(average_precision_score(y_test,y_pred[:,1]))

jb.dump(md,'lgb.pkl.z')
jb.dump(vectorizer_t,'title_vec.pkl.z')
jb.dump(vectorizer_s,'sinopse_vec.pkl.z')