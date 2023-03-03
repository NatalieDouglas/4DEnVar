import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
import sys
sys.path.insert(1, '/Users/nataliedouglas/Documents/Research/Reading University Work/4DEnVar')
from  scipy.stats import gaussian_kde as kde
import scipy.stats as stats
import scipy.optimize as spop
from scipy.linalg import sqrtm
import pandas as pd
import importlib

class fourDEnVar_engine:
    
    def __init__(self, Xb, hX, y, R, hxbar):   
            
        self.Xb=Xb
        self.hX=hX
        self.y=y
        self.R=R
        self.hxbar=hxbar
        print("shape of Xb: " + str(np.shape(self.Xb)))
        print("shape of hX: " + str(np.shape(self.hX)))
        print("shape of y: " + str(np.shape(self.y)))
        print("shape of R: " + str(np.shape(self.R)))
        print("shape of hxbar :" + str(np.shape(self.hxbar)))
        # incorporate tests for correct dimension compatibility
        
        self.n=np.shape(Xb)[0]
        self.nens=np.shape(Xb)[1]
        self.nobs=np.shape(y)[0]
        self.R_inv = inv(R) # temporary
        self.xbar = np.mean(Xb,1)
        self.scale = 1./np.sqrt(self.nens-1.)
        
        # EQUATION 7 in 4DEnVar notes:
        self.Xb_dash = self.scale*(Xb-self.xbar.reshape(self.n,1))
        print('shape of X_dash_b: ' + str(np.shape(self.Xb_dash)))
        # EQUATION 9 in 4DEnVar notes:
        self.Yb_dash = self.scale*(hX-self.hxbar.reshape(self.nobs,1))
        print('shape of Y_dash_b: ' + str(np.shape(self.Yb_dash)))
        
        self.fourDEnVar()
        self.fourDEnVar_sample_posterior()     
    
    # MINIMISATION PROCEDURE
    def fourDEnVar(self):

        minimiser = spop.minimize(self.fourDEnVar_cost_f, np.zeros((self.nens,1)), method='L-BFGS-B', jac=self.fourDEnVar_cost_df,                                          options={'gtol': 1e-16,'maxiter':100})
        # EQUATION 8 in 4DEnVar notes:
        self.xa=self.xbar+np.dot(self.Xb_dash, minimiser.x)  
        
        # ANALYTICAL SOLUTION
        #work1=np.dot(self.Yb_dash.T,np.dot(self.R_inv,self.Yb_dash))
        #work2=np.dot(self.Yb_dash.T,np.dot(self.R_inv,self.hx_bar-self.y))
        #w_analytical=-np.dot(inv(np.identity(self.nens)+work1),work2)
        work1=self.R+np.dot(self.Yb_dash,self.Yb_dash.T)
        work2=np.dot(self.Yb_dash.T,inv(work1))
        w_analytical=-np.dot(work2,self.hxbar-self.y)
        self.xa_analytical=self.xbar+np.dot(self.Xb_dash, w_analytical)  # this should be xb ??
        
    # COST FUNCTION
    def fourDEnVar_cost_f(self, w):

        bgrnd_term=np.dot(w.T,w)
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
        df=np.dot(self.Yb_dash.T,work2)
      
        return df
 
    # POSTERIOR ENSEMBLE
    def fourDEnVar_sample_posterior(self):

        work1 = np.dot(self.R_inv,self.Yb_dash)
        work2 = np.dot(self.Yb_dash.T,work1)+np.identity(self.nens)
        if np.allclose(work2,work2.T) and np.all(np.linalg.eigvals(work2) > 0):
            print('Xbdash*inv(I+Ybdash^T*Rinv*Ybdash)*Xbdash^T is symmetric and all eigenvalues are positive')
         
        work3 = np.linalg.cholesky(work2)
        #print('Lower triangular matrix from Cholesky decomposition:')
        #print(work3)
        
        # EQUATION 15 in 4DEnVar notes:
        Wa = inv(work3) # temporary
        # EQUATION 14 in 4DEnVar notes:
        Xa_dash = np.dot(self.Xb_dash,Wa)
        # EQUATION 13 in 4DEnVar notes:
        self.Pa = np.dot(Xa_dash,Xa_dash.T)
        # EQUATION 16 in 4DEnVar notes:
        self.Xa = (1./self.scale)*Xa_dash+self.xa.reshape(self.n,1)

        