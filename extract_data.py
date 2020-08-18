import pandas as pd
from bs4 import BeautifulSoup
import requests as rq
import re
import unidecode
import time
import tqdm 
import numpy as np
import logging
import googletrans

logger = logging.getLogger(__name__)


def get_data_from_imdb(urls,total):
       
    
    df = pd.DataFrame()
    for url in tqdm.tqdm(urls,total=len(urls),leave=True,position=0):
        for item in range(0,total,49):
            urll = url.format(itens=item)
            req = rq.get(url=urll,timeout=40)
            
            if (req.status_code == 200):
                page_html = req.text
                
                soup = BeautifulSoup(page_html,'html.parser')
                
                tags = soup.find_all('div',attrs={'class':'lister-item-content'})
                for t in tags:
                   try: 
                       
                       list_actor = []
                       link = 'https://www.imdb.com/'+t.find('a')['href']
                       title = t.find('a').get_text().strip()        
                       year = t.find('span',class_=re.compile(r'lister-item-year text-muted')).get_text().strip()
                       if t.find('p').find('span',class_=re.compile(r'certificate')):
                           certificate = t.find('p').find('span',class_=re.compile(r'certificate')).get_text().strip()
                       else:
                           certificate = np.NaN
                       if t.find('p').find('span',class_=re.compile(r'runtime')):
                           runtime = t.find('p').find('span',class_=re.compile(r'runtime')).get_text().strip()
                       else:
                           runtime = np.NaN
                       genre = t.find('p').find('span',{'class':'genre'}).get_text().strip()
                       if t.find('div',{'class':'ratings-bar'}):
                           if t.find('div',{'class':'ratings-bar'}).find('strong'):
                               rating = t.find('div',{'class':'ratings-bar'}).find('strong').get_text().strip()
                           else:
                               rating = np.NaN
                       else:
                           rating = np.NaN
                           
                       s = t.find_all('p',class_='text-muted')[1].get_text()
                       if s: 
                           sinopse = unidecode.unidecode(s.strip())
                           translator = googletrans.Translator()
                           
                           sinopse_br = translator.translate(sinopse,src='en',dest='pt').text
                                                 
                       if t.find('p',class_=re.compile(r'sort-num_votes-visible')):
                           votes = t.find('p',class_=re.compile(r'sort-num_votes-visible')).find('span',{'name':'nv'}).get_text().strip()
                       else:
                           votes = np.NaN
                           
                       stars = t.find_all('p',class_='')
                       for star in stars:
                           for actor in star.find_all('a'):
                               list_actor.append(actor.contents[0])
                           
                       
                       data = {'link':link,
                               'title': title,
                               'year': year,
                               'certificate': certificate,
                               'runtime':runtime,
                               'genre':genre,
                               'rating':rating,
                               'sinopse':sinopse,
                               'votes':votes,
                               'actors': ','.join(list_actor),
                               'sinopse_br':sinopse_br}
                        
                       df = df.append(data,ignore_index=True)
                   except Exception as e:
                       logger.exception('erro ao inserir: ', title)
                       continue
                
                time.sleep(1)
              
    df.to_csv('new_movies.csv',index=False)            

