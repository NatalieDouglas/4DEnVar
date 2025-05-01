import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
import sys
from  scipy.stats import gaussian_kde as kde
import scipy.stats as stats
import scipy.optimize as spop
from scipy.linalg import sqrtm
import pandas as pd
import importlib

class fourDEnVar_engine_gamma:
    
    def __init__(self, Xb, hX, y, R, hxbar,gamma):   
        
        self.Xb=Xb
        self.hX=hX
        self.y=y
        self.R=R
        self.hxbar=hxbar
        self.gamma=gamma

        self.n=np.shape(Xb)[0]
        self.nens=np.shape(Xb)[1]
        self.xbar = np.mean(Xb,1)
        
        self.nobs=len(y)
        self.R_inv = np.linalg.pinv(R)

        self.scale = 1./np.sqrt(self.nens-1.)
        
        # EQUATION 7 in 4DEnVar notes:
        self.Xb_dash = self.scale*(Xb-self.xbar.reshape(self.n,1))
        # EQUATION 9 in 4DEnVar notes:
        self.Yb_dash = self.scale*(hX-self.hxbar.reshape(self.nobs,1))
        
        self.fourDEnVar()
        self.fourDEnVar_sample_posterior() 
    
    # MINIMISATION PROCEDURE
    
    def fourDEnVar(self):

        minimiser = spop.minimize(self.fourDEnVar_cost_f, np.zeros((self.nens,1)), method='L-BFGS-B',                                                    jac=self.fourDEnVar_cost_df, options={'gtol': 1e-16,'maxiter':10000000},callback=self.callbackF)

        # EQUATION 8 in 4DEnVar notes:
        self.w=minimiser.x
        self.Jmin=minimiser.fun
        self.xa=self.xbar+np.dot(self.Xb_dash, self.w)  
        self.simobs=self.hxbar+np.dot(self.Yb_dash,self.w)
        
        # ANALYTICAL SOLUTION
        work1=np.dot(self.Yb_dash.T,np.dot(self.R_inv,self.Yb_dash))
        work2=np.dot(self.Yb_dash.T,np.dot(self.R_inv,self.hxbar-self.y))
        self.w_analytical=-np.dot(np.linalg.pinv(self.gamma*np.identity(self.nens)+work1),work2)
        self.xa_analytical=self.xbar+np.dot(self.Xb_dash, self.w_analytical)
        
        ##checks
        wgrad1=np.dot(self.Yb_dash,self.w_analytical)+self.hxbar-self.y
        wgrad2=np.dot(self.R_inv,wgrad1)
        wgrad=np.dot(self.Yb_dash.T,wgrad2)
        wcost=np.dot(wgrad1.T,wgrad2)

        
    # COST FUNCTION
    def fourDEnVar_cost_f(self, w):

        bgrnd_term=self.gamma*np.dot(w.T,w)
        
        work1=np.dot(self.Yb_dash,w)+self.hxbar-self.y
        work2=np.dot(self.R_inv,work1)
        obs_term=np.dot(work1.T,work2)
    
        # EQUATION 10 in 4DEnVar notes:
        J=0.5*bgrnd_term+0.5*obs_term
                                
        return J

    # GRADIENT FUNCTION
    def fourDEnVar_cost_df(self, w):

        work1=np.dot(self.Yb_dash,w)+self.hxbar-self.y
        work2=np.dot(self.R_inv, work1)
        
        # EQUATION 11 in 4DEnVar notes:
        df=self.gamma*w+np.dot(self.Yb_dash.T,work2)
      
        return df
 
    # POSTERIOR ENSEMBLE
    def fourDEnVar_sample_posterior(self):

        work1 = np.dot(self.R_inv,self.Yb_dash)
        work2 = np.dot(self.Yb_dash.T,work1)+self.gamma*np.identity(self.nens)
        work3 = np.linalg.cholesky(work2)
        
        # EQUATION 15 in 4DEnVar notes:
        Wa = np.linalg.pinv(work3) 
        # EQUATION 14 in 4DEnVar notes:
        Xa_dash = np.dot(self.Xb_dash,Wa)
        # EQUATION 13 in 4DEnVar notes:
        self.Pa = np.dot(Xa_dash,Xa_dash.T)
        # EQUATION 16 in 4DEnVar notes:
        self.Xa = (1./self.scale)*Xa_dash+self.xa.reshape(self.n,1)
        self.Xa_analytical = (1./self.scale)*Xa_dash+self.xa_analytical.reshape(self.n,1)


