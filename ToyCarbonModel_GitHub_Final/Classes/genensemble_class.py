import numpy as np
import functools
import sys
sys.path.insert(1, '/Users/nataliedouglas/Documents/Research/Reading University Work/ToyCarbonModel')

class genensemble_class:

    def __init__(self, nens, Xb, B_mat, val_bnds):   
         
        self.size_ens=nens
        self.Xb=Xb
        self.B_mat = B_mat
        self.val_bnds = val_bnds
        self.generate_x_ens()
    
    ## function to test the bounds and correct behaviour in trajectories
    def test_bnds(xbi, val_bnds):

        for xi in enumerate(xbi):
            if val_bnds[xi[0]][0] <= xi[1] <= val_bnds[xi[0]][1]:
                continue
            else:
                return False
        s0=0.01
        F0=1
        c0=xbi[0]*xbi[1]*xbi[3]*s0/(xbi[2]**2)
        c1=((xbi[0]*xbi[2]+xbi[1]*xbi[3])*s0-xbi[0]*xbi[1]*xbi[2]*xbi[3])/(xbi[2]**2)
        c2=(F0-xbi[0]*xbi[2]-xbi[1]*xbi[3]+s0)/xbi[2]
        c3=-1

        coeffs=[c3,c2,c1,c0]
        cubicroots=np.roots(coeffs)
        realroots = np.sort(cubicroots.real[abs(cubicroots.imag)<1e-5])
        if len(realroots)==3:
            if float(realroots[0]) < 0.5:
                return False    
        if len(realroots)==1:
            if float(realroots) < 0.5:
                return False
    
        return True

    ## generate restricted and unrestricted parameter ensembles
    def generate_x_ens(self):

        self.ens = []
        self.ens_full=[]
        #i = 0
        while len(self.ens) < self.size_ens:
            Xbi = np.random.multivariate_normal(self.Xb, self.B_mat)
            #i += 1
            if genensemble_class.test_bnds(Xbi, self.val_bnds) is True:
                self.ens.append(Xbi)
            else:
                continue
        
        for i in range(0,self.size_ens):
            Xbi = np.random.multivariate_normal(self.Xb, self.B_mat)
            self.ens_full.append(Xbi)
        
        self.ens=np.array(self.ens).T
        self.ens_full=np.array(self.ens_full).T
        
        return


