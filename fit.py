from microprediction import MicroWriter
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import random 
import time
from pprint import pprint
import warnings
warnings.filterwarnings('ignore')
from copulas.multivariate import GaussianMultivariate
import pandas as pd


# Grab the Github secrets
# For this script to work you need to create four separate GitHub secrets
# called WRITE_KEY_1 WRITE_KEY_2 WRITE_KEY_3 and WRITE_KEY_4
import os 
WRITE_KEYS = [ os.environ.get('WRITE_KEY_'+str(i+1)) for i in range(4) ]      


print('Copula syndicate is firing up.')
for write_key in WRITE_KEYS:
    animal = MicroWriter.animal_from_key(write_key)  
    print(animal)
    
REPO = 'https://github.com/microprediction/microactors-plots/blob/master/fit.py'  # <--- Change your username


# Get historical data, fit a copula, and submit 

def fit_and_sample(lagged_zvalues:[[float]],num:int, copula=None):
    """ Example of fitting a copula function, and sampling
           lagged_zvalues: [ [z1,z2,z3] ]  distributed N(0,1) margins, roughly
           copula : Something from https://pypi.org/project/copulas/
           returns: [ [z1, z2, z3] ]  representative sample
    """
    # This is the part you'll want to change. 
    # Remark 1: It's lazy to just sample synthetic data
    # Some more evenly spaced sampling would be preferable. 
    # Remark 2: Any multivariate density estimation could go here. 
    # Remark 3: If you want to literally fit to a Copula (i.e. roughly uniform margins)
    # then you might want to use mw.get_lagged_copulas(name=name, count= 5000) instead
    #
    # See https://www.microprediction.com/blog/lottery for discussion of why evenly 
    # spaced samples are likely to serve you better. 

    df = pd.DataFrame(data=lagged_zvalues)
    if copula is None:
        copula = GaussianMultivariate() # <--- Change the Copula to something else?
    copula.fit(df)
    synthetic = copula.sample(4*num)
    return synthetic.values.tolist()


if __name__ == "__main__":
    mw = MicroWriter(write_key=WRITE_KEY)
    mw.set_repository(REPO) # Just polite

    NAMES = [ n for n in mw.get_stream_names() if 'z2~' in n or 'z3~' in n ]
    for _ in range(5):
        name = random.choice(NAMES)
        for delay in [ mw.DELAYS[0], mw.DELAYS[-1]]:
            lagged_zvalues = mw.get_lagged_zvalues(name=name, count= 5000)
            if len(lagged_zvalues)>20:
                num = mw.num_predictions
                four = len(WRITE_KEYS)
                zvalues = fit_and_sample(lagged_zvalues=lagged_zvalues, num=num*four)
                pprint((name, delay))
                try:
                    responses = list()
                    for j, write_key = enumerate(WRITE_KEYS)
                        zvalues_j = zvalues[j*num:(j+1)*num]
                        assert len(zvalues_j)==num
                        responses.append( mw.submit_zvalues(name=name, zvalues=zvalues_j, delay=delay ) )
                    pprint(responses)
                except Exception as e:
                    print(e)
