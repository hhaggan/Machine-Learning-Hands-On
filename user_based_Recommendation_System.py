import pandas as pd
import numpy as np 

df = pd.DataFrame(data={'Zin': [7,0,3,2,5], 
                        'Pinot Noir': [6,7,3,2,6],
                        "Chard": [7,6,3,1,7],
                        "Merlot": [4,4,1,3,2],
                        'Cab': [5,3,1,7,3],
                        "Pinot Gris": [4,4,5,4,3]})

df.index = ['Yuri', 'Steve', 'Gary', 'Qurat', 'Brigid']

from sklearn.metrics.pairwise import *

#np.around(cosine_similarity(df), 2)
np.around(cosine_distances(df), 2)