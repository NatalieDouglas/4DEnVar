import numpy as np
import pandas as pd
import sys
sys.path.insert(1, '/Users/nataliedouglas/Documents/Research/Reading University Work/4DEnVar')
from matplotlib import pyplot as plt
import fourDEnVar_engine
import importlib
importlib.reload(fourDEnVar_engine)


class linearModel:

    def __init__(self, coefs):
        self.coefs=coefs # define coefficients for quadratic

    def evalpol(self, x):
        return np.polyval(self.coefs,x) # evaluate quadratic with given coefficients at given x values

class linearModelEnsemble:

    def __init__(self, coefs_truth, coefs_prior, coefs_ens, obs):

        self.truth=linearModel(coefs_truth) 
        self.prior=linearModel(coefs_prior) 

        self.coefs_prior=coefs_prior 
        self.coefs_ens=coefs_ens
        
        self.obs=obs
              
        self.range=(0,1)
        self.rand_obs_x=False # when false this uniformly discretises x axis for observation resolution
        
        # run functions below
        self.gen_prior_ensemble()
        self.gen_obs()

    # generate y coordinates for ensemble of coefficients   
    def gen_prior_ensemble(self):
        self.ensemble=[]
        for n in range(np.shape(self.coefs_ens)[1]):
            coefs=self.coefs_ens[:,n]
            self.ensemble.append(linearModel(coefs))                
    
    # define observations at specified x values
    def gen_obs(self):
        # random step size
        if self.rand_obs_x:
            self.obs_x=np.random.rand(self.nobs)
            self.obs_x=self.obs_x*(self.range[1]-self.range[0])+self.range[0]
        # uniform step size
        else:
            a=self.range[0]
            b=self.range[1]
            step=(b-a)/np.shape(self.obs)[0]
            self.obs_x=np.arange(a+step/2.,b,step)
            
        self.obs_y=self.obs
    
    # plot quadratics
    def plot(self,filename=None, analysis=None, check=None):
        a=self.range[0]
        b=self.range[1]
        # discretise x domain uniformly
        x=np.arange(a,b,(b-a)/100.)
        
        # plot ensemble
        for n in range(np.shape(self.obs)[0]):
            plt.plot(x,self.ensemble[n].evalpol(x),'k',alpha=0.2)
    
        # add prior mean, truth and observations
        plt.plot(x,self.prior.evalpol(x),'b',label='prior')
        plt.plot(x,self.truth.evalpol(x),'r',label='truth + obs')
        plt.plot(self.obs_x,self.obs_y,'.r')
        
        # plot analysis
        if analysis is not None:
            l=linearModel(analysis)
            plt.plot(x,l.evalpol(x),'g',label='analysis')
        if check is not None:
            l_check=linearModel(check)
            plt.plot(x,l_check.evalpol(x),'y',label='analytical analysis')
    
        plt.xlim(self.range)
        plt.ylim((self.truth.evalpol(a),self.truth.evalpol(b)))
        plt.legend()
        plt.show()
        plt.savefig(filename)

            
if __name__=="__main__":
    
    # import 4DEnVar attributes
    xb=np.genfromtxt("Xb.dat")
    hx=np.genfromtxt("HX.dat")
    y=np.genfromtxt("y.dat")
    R=np.genfromtxt("R.dat")
    hx_bar=np.genfromtxt("Hxbar.dat")
    
    # run 4DEnVar
    x=fourDEnVar_engine.fourDEnVar_engine(xb, hx, y, R, hx_bar)  
    print("minimisation analysis: xa = : " + str(x.xa))
    print("analytical analysis: xa_check = : " + str(x.xa_check))
    
    truth=[2.,1.1,0.] # true parameters/coefficients
    xb_bar=np.mean(xb,1) # prior from ensemble mean
    
    # run linearModelEnsemble function above
    l=linearModelEnsemble(truth, xb_bar, xb, y)
    #l.plot(filename="linear_example.png",analysis=x.xa, check=x.xa_check)
    l.plot(filename="linear_example.png",analysis=x.xa, check=None)
