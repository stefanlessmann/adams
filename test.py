import numpy as np
import pandas as pd

appStore = pd.read_csv('./data/AppleStore.csv')
papp = pd.read_csv('./data/PrepApp.csv',index_col=False,sep='\t', encoding='utf-8')


papp.head()
