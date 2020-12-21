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
from copulas.visualization import scatter_2d
import os


os.environ['WRITE_KEY_1']='baf7e5692237cc98b10dd9646b464bf0'
os.environ['WRITE_KEY_2']='f8ca8202168445c9f42b8618a06bbe1a'
os.environ['WRITE_KEY_3']='028f5b439d75ac1ca02c59ed63e12b7d'
os.environ['WRITE_KEY_4']='a6c968a9b1f16693405353231e8f1488'




# 1. Grab the Github secrets
# For this script to work you need to create four separate GitHub secrets
# called WRITE_KEY_1 WRITE_KEY_2 WRITE_KEY_3 and WRITE_KEY_4
# The idea is that this way you get 900 samples instead of 225
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
def fit_and_sample(lagged_zvalues:[[float]],num:int, copula=None, fig_file=None, labels=None ):
    """ Example of fitting a copula function, and sampling
    
           lagged_zvalues: [ [z1,z2,z3] ]  Data with roughly N(0,1) margins
           copula : 
           returns: [ [z1, z2, z3] ]  representative sample
           labels: [str]  axis labels
           
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
    print('Fit done')
    synthetic = copula.sample(4*num)  # Again, see remarks above
    print('Sample generated')
    synth = synthetic.values.tolist()
    dim = len(synth[0])
    if dim==3 and fig_file is not None:
        print('Saving to '+fig_file)
        plot_3d(real=real, synth=synthetic, fig_file=fig_file, labels=labels)
    return synth


def scatter_3d(data, columns=None, fig=None, title=None, position=None, labels=None):
    """Plot 3 dimensional data in a scatter plot."""
    fig = fig or plt.figure()
    position = position or 111

    ax = fig.add_subplot(position, projection='3d')
    ax.scatter(*(
        data[column]
        for column in columns or data.columns
    ))
    if title:
        ax.set_title(title)
        ax.title.set_position([.5, 1.05])
    if labels:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])

    return ax

def plot_3d(real, synth, fig_file, columns=None, figsize=(20, 8), labels=None ):
    """ Create and store comparision plot """
    columns = columns or real.columns
    fig = plt.figure(figsize=figsize)
    scatter_3d(real[columns], fig=fig, title='Real Data', position=121, labels=labels)
    scatter_3d(synth[columns], fig=fig, title='Synthetic Data', position=122, labels=labels)
    plt.tight_layout()
    plt.savefig(fig_file)
    
    
if __name__ == "__main__":
    mws = [ MicroWriter(write_key=write_key) for write_key in WRITE_KEYS ]
    for mw in mws:
        mw.set_repository(REPO)
    mw0 = mws[0] # Doesn't matter which one
    NAMES = [ n for n in mw0.get_stream_names() if 'z3~' in n ]
    for _ in range(5):
        name = random.choice(NAMES)
        labels = name.split('~')[1:-1]
        lagged_zvalues = mw0.get_lagged_zvalues(name=name, count=5000)
        if len(lagged_zvalues) > 20:
            for delay in [ mw0.DELAYS[0], mw0.DELAYS[-1]]:
                num = mw0.num_predictions
                four = len(WRITE_KEYS)
                fig_file = PLOTS_PATH + os.path.sep + name.replace('.json','')+'_'+str(delay) +'_'+ VINE_TYPE.lower()+'.png'
                pprint((name, delay, len(lagged_zvalues)))
                zvalues = fit_and_sample(lagged_zvalues=lagged_zvalues, num=num*four, fig_file=fig_file,labels=labels)
                print('Syndicate submission starting')
                try:
                    # Split the samples up amongst the syndicate
                    # This would be more effective if the samples were not random :-)
                    responses = list()
                    for j, mw in enumerate(mws):
                        zvalues_j = zvalues[j*num:(j+1)*num]
                        assert len(zvalues_j)==num
                        responses.append( mw.submit_zvalues(name=name, zvalues=zvalues_j, delay=delay ) )
                    pprint(responses)
                except Exception as e:
                    print(e)
                print('Syndicate submission finished')
        else:
            print(name+' history too short ')
