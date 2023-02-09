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
    
    def __init__(self, xb, hx, y, R, hx_bar):   
            
        self.xb=xb
        self.hx=hx
        self.y=y
        self.R=R
        self.hx_bar=hx_bar
        print("shape of xb: " + str(np.shape(self.xb)))
        print("shape of hX: " + str(np.shape(self.hx)))
        print("shape of y: " + str(np.shape(self.y)))
        print("shape of R: " + str(np.shape(self.R)))
        print("shape of hXbar :" + str(np.shape(self.hx_bar)))
        # incorporate tests for correct dimension compatibility
        
        self.n=np.shape(xb)[0]
        self.nens=np.shape(xb)[1]
        self.nobs=np.shape(y)[0]
        self.R_inv = inv(R) # temporary
        self.xb_bar = np.mean(xb,1)
        self.scale = 1./np.sqrt(self.nens-1.)
        
        # EQUATION 7 in 4DEnVar notes:
        self.Xb_dash = self.scale*(xb-self.xb_bar.reshape(self.n,1))
        print('shape of X_dash_b: ' + str(np.shape(self.Xb_dash)))
        # EQUATION 9 in 4DEnVar notes:
        self.Yb_dash = self.scale*(hx-hx_bar.reshape(self.nobs,1))
        print('shape of Y_dash_b: ' + str(np.shape(self.Yb_dash)))
        
        self.fourDEnVar()
        self.fourDEnVar_sample_posterior()     
    
    # MINIMISATION PROCEDURE
    def fourDEnVar(self):

        minimiser = spop.minimize(self.fourDEnVar_cost_f, np.zeros((self.nens,1)), method='L-BFGS-B', jac=self.fourDEnVar_cost_df,                                          options={'gtol': 1e-16,'maxiter':100})
        # EQUATION 8 in 4DEnVar notes:
        self.xa=self.xb_bar+np.dot(self.Xb_dash, minimiser.x)  
        
        # CHECK
        work1=np.dot(self.Yb_dash.T,np.dot(self.R_inv,self.Yb_dash))
        work2=np.dot(self.Yb_dash.T,np.dot(self.R_inv,self.hx_bar-self.y))
        w_check=-np.dot(inv(np.identity(self.nens)+work1),work2)
        self.xa_check=self.xb_bar+np.dot(self.Xb_dash, w_check)  
        
    # COST FUNCTION
    def fourDEnVar_cost_f(self, w):

        bgrnd_term=np.dot(w.T,w)
        work1=np.dot(self.Yb_dash,w)+self.hx_bar-self.y
        work2=np.dot(self.R_inv,work1)
        obs_term=np.dot(work1.T,work2)
    
        # EQUATION 10 in 4DEnVar notes:
        J=0.5*bgrnd_term+0.5*obs_term
                                
        return J

    # GRADIENT FUNCTION
    def fourDEnVar_cost_df(self, w):

        work1=np.dot(self.Yb_dash,w)+self.hx_bar-self.y
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

        