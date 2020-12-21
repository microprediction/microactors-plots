from microprediction import MicroWriter
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import random 
import time
from pprint import pprint
import warnings
warnings.filterwarnings('ignore')
from copulas.multivariate import VineCopula
import pandas as pd
from copulas.visualization import scatter_2d, scatter_3d


# 1. Grab the Github secrets
# For this script to work you need to create four separate GitHub secrets
# called WRITE_KEY_1 WRITE_KEY_2 WRITE_KEY_3 and WRITE_KEY_4
# The idea is that this way you get 900 samples instead of 225
import os 
WRITE_KEYS = [ os.environ.get('WRITE_KEY_'+str(i+1)) for i in range(4) ]   
print('Copula syndicate is firing up.')
for write_key in WRITE_KEYS:
    animal = MicroWriter.animal_from_key(write_key)  
    print(animal)
    
    
# 2. Pick a copula
VINES = ['center','regular','direct'] # See https://sdv.dev/Copulas/tutorials/03_Multivariate_Distributions.html#Vine-Copulas
VINE_TYPE = random.choice(VINES)       # Perhaps you want to fix this choice. This way we get lots of plots. 

# 3. (Optional) Set the URL of your repo so that others can learn from it     
REPO = 'https://github.com/microprediction/microactors-plots/blob/master/fit.py'  # <--- Change your username

PLOTS_PATH = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + 'gallery'


# 4. Create your fitting function 
def fit_and_sample(lagged_zvalues:[[float]],num:int, copula=None, fig_file=None ):
    """ Example of fitting a copula function, and sampling
    
           lagged_zvalues: [ [z1,z2,z3] ]  Data with roughly N(0,1) margins
           copula : 
           returns: [ [z1, z2, z3] ]  representative sample
           
    """
    # This is the part where there's plenty of room for improvement 
    # Remark 1: It's lazy to just sample synthetic data
    #           Some more evenly spaced sampling would be preferable. 
    #           See https://www.microprediction.com/blog/lottery for discussion of why evenly 
    #           spaced samples are likely to serve you better. 
    # Remark 2: Any multivariate density estimation could go here. 
    # Remark 3: If you want to literally fit to a Copula (i.e. roughly uniform margins)
    # then you might want to use mw.get_lagged_copulas(name=name, count= 5000) instead
    

    real = pd.DataFrame(data=lagged_zvalues)
    if copula is None:
        copula = VineCopula(VINE_TYPE) 
    copula.fit(real)
    synthetic = copula.sample(4*num)  # Again, see remarks above
    synth = synthetic.values.tolist()
    dim = len(synth[0])
    if dim==3 and fig_file is not None:
        plot_3d(real=real, synth=synth, fig_file=fig_file)
    return synth

def plot_3d(real, synth, fig_file, columns=None, figsize=(10, 4)):
    """ Create and store comparision plot """
    columns = columns or real.columns
    fig = plt.figure(figsize=figsize)
    scatter_3d(real[columns], fig=fig, title='Real Data', position=121)
    scatter_3d(synth[columns], fig=fig, title='Synthetic Data', position=122)
    plt.tight_layout()
    plt.savefig(fig_file)
    
    
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
                fig_file = PLOTS_PATH + os.path.sep + name.replace('.json','')+'_'+str(delay) +'_'+ VINE_TYPE.lower()+'.png'
                zvalues = fit_and_sample(lagged_zvalues=lagged_zvalues, num=num*four, fig_file=fig_file)
                pprint((name, delay))
                try:
                    # Split the samples up amongst the syndicate
                    # This would be more effective if the samples were not random :-)
                    responses = list()
                    for j, write_key in enumerate(WRITE_KEYS):
                        zvalues_j = zvalues[j*num:(j+1)*num]
                        assert len(zvalues_j)==num
                        responses.append( mw.submit_zvalues(name=name, zvalues=zvalues_j, delay=delay ) )
                    pprint(responses)
                except Exception as e:
                    print(e)
