import numpy as np
import importlib
import modelruns_class
importlib.reload(modelruns_class)

class genDAfeatures_class:

    def __init__(self, initstate, tf, deltat, forcing, ensemble):   
         
        self.initstate=initstate
        self.tf=tf
        self.deltat=deltat
        self.forcing=forcing
        self.x_ens=ensemble
        if len(np.shape(self.x_ens))==1:
            self.nens=np.shape(self.x_ens)
            self.x_bar=np.mean(self.x_ens)
        else:    
            self.nens=np.shape(self.x_ens)[1]
            self.x_bar=np.mean(self.x_ens,1)
        self.getfeatures()
    
    ## generates hxbar and hX for a given ensemble
    def getfeatures(self):

        MR = modelruns_class.modelruns_class(self.initstate, self.tf, self.deltat, self.forcing, self.x_bar)
        self.hxbar_mat = MR.xf
        self.hxbar = np.concatenate(self.hxbar_mat.T,axis=0)
        
        self.hX = np.zeros((self.tf*2+2,self.nens))
        self.hX_mat = np.zeros((2,self.tf+1,self.nens))
        for i in range(0,self.nens):
            MR = modelruns_class.modelruns_class(self.initstate, self.tf, self.deltat, self.forcing, self.x_ens[:,i])
            self.hX_mat[:,:,i]=MR.xf
            self.hX[:,i]=np.concatenate(self.hX_mat[:,:,i].T,axis=0)                     
        
        return


