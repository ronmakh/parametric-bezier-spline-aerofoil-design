import numpy as np
import matplotlib.pyplot as pl
from scipy.optimize import minimize

def cornerspeed(V, radius, Cl, M, S):
    rho=1.2;
    L=0.5*rho*V**2*S*Cl;
    mu=2;
    Ff=mu*(M*9.81+L);
    v1=np.sqrt(Ff*radius/M);
    diff=(V-v1)**2;
    return diff

def simulator(X, plotting):

    from lapsim import cornerspeed
    
    clA = X[0]
    cdA = X[1]

    seg = np.array([0.16, 0.12, 0.33, 0.09, 0.1+0.11+0.1+0.25, 0.05+0.06+0.03+0.06+0.13+0.09+0.06]) * 1602;
    ang = np.array([90, 15, 75, 40, 140]);
    rad = np.array([6, 20, 10, 15, 3]);
    rad = rad*2;
    ab=-2.5;
    S=1.0;
    cl=clA/S;
    cd=cdA/S;
    rho=1.2;
    T=100;
    F=T;
    M=25;
    
    vcorner = np.zeros([5,1])
    scorner = np.zeros([5,1])
    tcorner = np.zeros([5,1])
    tbrake = np.zeros([5,1])
    
    for I in range(0,5):
        opt_out=minimize(cornerspeed,10,args=(rad[I], cl, M, S))
        vcorner[I] = opt_out.x[0]
        scorner[I] = (ang[I]*np.pi/180)*rad[I];
        tcorner[I]=scorner[I]/vcorner[I];
    
        i = 1
        t = 0.1
        s = np.array([0.0])
        time = np.array([0.0])
    
        if I == 0:
            v = np.array([0.0])
        else:
            v = np.array(vcorner[I-1])
    
        D=0.5*rho*v[-1]**2*S*cd;
        F=T-D;
        stot=0;
        while stot<seg[I]:
            time= np.append(time, time[-1]+t)
            a=F/M
            v = np.append(v, v[-1]+a*t)
            D=0.5*rho*v[-1]**2*S*cd
            F=T-D
            s=np.append(s, s[-1]+v[-1]*t+0.5*a*t**2)
    
            if v[-1]>vcorner[I]:
                tbrake[I]=(vcorner[I]-v[-1])/ab;
                sbrake=v[-1]*tbrake[I]+0.5*ab*tbrake[I]**2;
                stot=s[-1]+sbrake;
            elif ((v[-1]<vcorner[I]) & (s[-1]>seg[I])):
                stot=s[-1]
                tbrake[I]=0
                vcorner[I]=v[-1]
                scorner[I]=(ang[I]*np.pi/180)*rad[I]
                tcorner[I]=scorner[I]/vcorner[I]
            i = i+1
        if I==0:    
            Time=np.concatenate((time, time[-1]+tbrake[I], time[-1]+tbrake[I]+tcorner[I]))
            Speed=np.concatenate((v, vcorner[I], vcorner[I]))
        if I>0:
            time=time+Time[-1]
            Time=np.concatenate((Time, time, time[-1]+tbrake[I], time[-1]+tbrake[I]+tcorner[I]))
            Speed=np.concatenate((Speed, v, vcorner[I], vcorner[I]))
    i=1;
    t=0.1;
    s=np.array([0.0]);
    time=np.array([0.0]);
    v=vcorner[I];
    D=0.5*rho*v[-1]**2*S*cd;
    F=T-D;

    while s[-1]<seg[I+1]:
        time = np.append(time, time[-1]+t);
        a=F/M
        v = np.append(v, v[-1]+a*t)
        D=0.5*rho*v[-1]**2*S*cd
        F=T-D
        s=np.append(s, s[-1]+v[-1]*t+0.5*a*t**2)
        i=i+1

    time=time+Time[-1];
    Time=np.append(Time, time);
    Speed=np.append(Speed, v);
    LapTime=Time[-1];
    
    if plotting == 1:
        pl.plot(Time,Speed)
        pl.xlabel('time [sec]')
        pl.ylabel('speed [m/s]')
        pl.show()
        
    return LapTime
