import numpy as np
import functools

class modelruns_class:

    def __init__(self, init, time, timestep, forcing, params):   
            
        self.x0=init
        self.tf=time
        self.deltat=timestep
        self.F=forcing
        self.params=params
        self.modelrun()

    ## Integrate model with RK4
    def rungekutta4(Xold, deltat, F, modelf):

        rk1 = modelf(Xold, F)
        rk2 = modelf(Xold + (1 / 2.0) * deltat * rk1, F)
        rk3 = modelf(Xold + (1 / 2.0) * deltat * rk2, F)
        rk4 = modelf(Xold + deltat * rk3, F)
        deltax = deltat * (rk1 + 2 * rk2 + 2 * rk3 + rk4) / 6.0
        return deltax

    ## Function to determine the differential equations for the toy carbon model
    def f(params, x, Ft):
    
        if np.all(params)==None:
            p1 = 1
            p2 = 1
            k1 = 0.2
            k2 = 0.1
        else:
            p1, p2, k1, k2 = params
     
        k = np.empty_like(x)
        k.fill(np.nan)
        k[0] = Ft*x[0]*x[1]/((x[0]+p1)*(x[1]+p2))+0.01-k1*x[0]
        k[1] = k1*x[0]-k2*x[1]
    
        return k
    
    ## Run the toy carbon model
    def modelrun(self):
        nx = len(self.x0)
        nt = int(self.tf / self.deltat) + 1
        self.xf = np.empty((nx, nt), order="F")
        self.xf.fill(np.nan)

        model = functools.partial(modelruns_class.f, self.params)

        self.xf[:, 0] = self.x0[:]

        for time in range(nt - 1):
            self.xf[:, time + 1] = self.xf[:, time] + modelruns_class.rungekutta4(self.xf[:, time], self.deltat, self.F[time], model)
        return self.xf

