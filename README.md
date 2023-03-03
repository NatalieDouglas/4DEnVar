# 4DEnVar
Notes and Python code for implementing the 4DEnVar Data Assimilation method with examples of use

## Summary
In the main branch of this repository you will find a PDF that explains the mathematics behind the 4DEnVar data assimilation method. You will also find   
two .py files: fourDEnVar.py and fourDEnVar_engine.py - the former to import the required data files to run 4DEnVar and the latter to apply the method as detailed in the notes. The required data files are:

Xb.dat - an array of the ensemble of background state vectors (with dimensions state by ensemble size)

hX.dat - an array of simulated observations, the ensemble mapped to observation space (with dimensions observations by ensemble size)

y.dat - a vector of the observations

R.dat - the observation error covariance matrix (with dimensions observations by observations)

hxbar.dat - a vector of the background ensemble mean mapped to observation space

You may need to tweak the code in fourDEnVar.py if your data files are not in .dat format.
    
## Linear Tests
The 'linear tests' folder contains all of the relevant files required to implement 4DEnVar for yourself. This includes all data files as listed above, a .py file which imports fourDEnVar_engine.py as a module and creates a plot to demonstrate the use of 4DEnVar. Please note that you will have to amend code line 4 in genLinear.py to point to the directory where you will store the fourDEnVar_engine.py file.

Running the genLinear.py file should produce the following image:
![linear_example](https://user-images.githubusercontent.com/93133873/216400849-c8fd1094-3672-4754-9df7-47f96f8c0668.png)

The state vector in this example is given by the three coefficients of a quadratic (the model is linear in the coefficients). 


## Still to come
I am currently working on some zero order tests that allows one to explore the prior and posterior distributions when implementing 4DEnVar. These will be uploaded shortly. It is likely that the notes and code will be updated from time to time (with dimensionality checks and revised inverse techniques, for example) so it is recommended that you 'watch' this repository for updates.
