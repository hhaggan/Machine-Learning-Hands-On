import pandas as pd
import numpy as np
from scipy import spatial
from sklearn.metrics.pairwise import *

df = pd.DataFrame(data={'Zin': [7,0,3,2,5], 
                        'Pinot Noir': [6,7,3,2,6],
                        "Chard": [7,6,3,1,7],
                        "Merlot": [4,4,1,3,2],
                        'Cab': [5,3,1,7,3],
                        "Pinot Gris": [4,4,5,4,3]})

df.index = ['Yuri', 'Steve', 'Gary', 'Qurat', 'Brigid']

#Cosine
#np.around(cosine_similarity(df), 2)
np.around(cosine_distances([df.iloc[1]], [df.iloc[2]]), 3)

spatial.distance.cosine(df.iloc[1], df.iloc[2])

#Euclidian
spatial.distance.euclidean(df.iloc[1], df.iloc[2])

np.around(euclidean_distances([df.iloc[1]], [df.iloc[2]]), 3)

#Jaccard Similarity
spatial.distance.jaccard(df.iloc[1], df.iloc[2])

