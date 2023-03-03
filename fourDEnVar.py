import numpy as np
import pandas as pd
import importlib
import fourDEnVar_engine
importlib.reload(fourDEnVar_engine)

if __name__=="__main__":

    Xb=np.genfromtxt("Xb.dat")
    hX=np.genfromtxt("hX.dat")
    y=np.genfromtxt("y.dat")
    R=np.genfromtxt("R.dat")
    hx_bar=np.genfromtxt("Hxbar.dat")
    
    x=fourDEnVar_engine.fourDEnVar_engine(Xb, hX, y, R, hx_bar)
    xa=x.xa
    Xa=x.X_a

    print(xa)
    print(Xa)
