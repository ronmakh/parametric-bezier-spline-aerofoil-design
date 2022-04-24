import matplotlib.pyplot as pl
import numpy as np
from scipy.optimize import fsolve

def choose(n, k):
    if 0 <= k <= n:
        p = 1
        for t in range(0,min(k, n - k),1):
            p = (p * (n - t)) // (t + 1)
        return p
    else:
        return 0

def mls(x, X, y, sigma): #1D quadratic moving least squares
    N = max(np.shape(y))
    weights = np.zeros(N)
    A = np.zeros([N, 3])
    A[:, 0] = X**2
    A[:, 1] = X
    A[:, 2] = np.ones([1, N])
    for i in range(0, N):
        weights[i] = np.exp(-np.sum((x - X[i])**2) / (2 * sigma))
    W = np.diag(weights)
    a = np.linalg.lstsq(np.dot(np.dot(A.conj().T, W), A), np.dot(np.dot(A.conj().T, W), y) )
    f = a[0][0] * x**2 + a[0][1] * x + a[0][2]
    return f

def mls_error(sigma, X, y): #1D quadratic moving least squares cross-validation
    y_test = np.zeros(len(y))
    error = np.zeros(len(y))
    for i in range(0,len(y)):
        y_test[i] = mls(X[i], np.append(X[0: i], X[i+1:-1]), np.append(y[0:i], y[i+1:-1]), sigma)
        error[i] = (y[i]-y_test[i])**2
    sum_error = sum(error)
    return sum_error

def mls_curve_fit(w_array,cd,x):
    sigma_best = fsolve(mls_error,0.5,args=(w_array,cd)) #fit moving least squares
    w_fine = np.linspace(np.min(w_array) ,np.max(w_array) ,101)
    y_pred= np.zeros(101)
    for i in range(0,101):
        y_pred[i] = mls(w_fine[i], w_array, cd, sigma_best)   
    pl.plot(w_array, cd, 'o', label='data points')
    pl.plot(w_fine, y_pred, label='MLS fit')
    pl.legend(bbox_to_anchor=(1.05, 1), loc=2)
    f = mls(x, w_array, cd, sigma_best)
    return f

def vortex_panel(pointsDef,alpha_deg,plot):
    xb=pointsDef[:,0]
    yb=pointsDef[:,1]
    npanel=np.shape(xb)[0]-1
    xc=np.zeros(npanel)
    yc=np.zeros(npanel)
    ds=np.zeros(npanel)
    dx=np.zeros(npanel)
    dy=np.zeros(npanel)
    theta=np.zeros(npanel)
    InfluenceMat=np.zeros((npanel+1,npanel+1))
    TangentialMat=np.zeros((npanel,npanel+1))
    R=np.zeros((npanel+1))
    psi=np.zeros((npanel+1))
    U=1;
    alpha=alpha_deg*np.pi/180.0;
    for i in range(0,npanel):
        xc[i]=(xb[i]+xb[i+1])/2.0
        yc[i]=(yb[i]+yb[i+1])/2.0
        ds[i]=((xb[i+1]-xb[i])**2+(yb[i+1]-yb[i])**2)**0.5
        dx[i]=xb[i+1]-xb[i]
        dy[i]=yb[i+1]-yb[i]
        if xc[i]<xb[i]:
            theta[i]=-np.arcsin((yb[i+1]-yb[i])/ds[i])
        else:
            theta[i]=np.arcsin((yb[i+1]-yb[i])/ds[i])-np.pi
    for i in range(0,npanel):
        for j in range(0,npanel):
            S=ds[j];
            sinij=np.sin(theta[i]-theta[j])
            cosij=np.cos(theta[i]-theta[j])
            sinji=np.sin(theta[j]-theta[i])
            cosji=np.cos(theta[j]-theta[i])
            # work in panel frame of ref
            xt=xc[i]-xb[j]
            yt=yc[i]-yb[j]
            # rotate
            xpc=xt*np.cos(theta[j])+yt*np.sin(theta[j])
            ypc=-xt*np.sin(theta[j])+yt*np.cos(theta[j])
        
            xt=xb[j+1]-xb[j]
            yt=yb[j+1]-yb[j]
            # rotate
            xpc2=xt*np.cos(theta[j])+yt*np.sin(theta[j])
            
            R1=(xpc**2+ypc**2)**0.5
            R2=((xpc-xpc2)**2+ypc**2)**0.5
            B1=np.arctan(ypc/xpc)
            B2=np.arctan(ypc/(xpc-xpc2))
            
            if ypc<0 and xpc<0:
                B1=B1-np.pi
            elif ypc>0 and xpc<0:
                B1=B1+np.pi
            if ypc<0 and (xpc-xpc2)<0:
                B2=B2-np.pi
            elif ypc>0 and (xpc-xpc2)<0:
                B2=B2+np.pi           
            B=B2-B1
            Ustar=-np.log(R2/R1)*(1/(2.0*np.pi))
            Vstar=B*(1/(2.0*np.pi))
            Ustar_v=Vstar;
            Vstar_v=-Ustar;
            if i==j:
                Ustar=0.0
                Vstar=0.5
                Ustar_v=0.5;
                Vstar_v=0;
            InfluenceMat[i,j]=-sinij*Ustar+cosij*Vstar
            InfluenceMat[i,npanel]=InfluenceMat[i,npanel]-sinij*Ustar_v+cosij*Vstar_v
            if i==npanel-1 or i==0:
                InfluenceMat[npanel,j]=InfluenceMat[npanel,j]+cosji*Ustar-sinji*Vstar
                InfluenceMat[npanel,npanel]=InfluenceMat[npanel,npanel]+cosji*Ustar_v-sinji*Vstar_v             
           
            TangentialMat[i,j]=cosji*Ustar-sinji*Vstar
            TangentialMat[i,npanel]=TangentialMat[i,npanel]+cosji*Ustar_v-sinji*Vstar_v
            
    R[0:-1]=U*np.sin(theta-alpha)
    R[-1]=-U*np.cos(theta[0]-alpha) - U*np.cos(theta[-1]-alpha)
    q, r=np.linalg.qr(InfluenceMat)
    p=np.dot(q.T,R)
    psi=np.dot(np.linalg.inv(r), p)
    vt=(U*np.cos(theta-alpha))+np.dot(TangentialMat,psi)
    cp=1.0-(vt/U)**2
    cx=sum(cp*dy)
    cy=sum(cp*dx)
    cl=-(cy*np.cos(alpha)-cx*np.sin(alpha))
    if plot==1:
        pl.plot(xc,-cp,'b')
        pl.plot(xc,yc*10,'k')
        pl.xlabel('x/c')
        pl.ylabel(r'$-C_P$, thickness $\times 10$')
        pl.xlim(-0.1, 1.1)
    return cl,cp,xc,yc,dy,ds,theta,vt

def hfromlambda(lam):
    if lam>0:
        H=2.61-3.75*lam+5.24*lam**2
    else:
        H=2.088+0.0731/(lam+0.14);
    return H

def h1fromh(H):
    if H<1.1:
        H1=10;
    elif H<1.6:
        H1=3.3+0.8234*(H-1.1)**(-1.287);
    else:
        H1=3.3+1.5501*(H-0.6778)**(-3.064);
    return H1

def nprexcr(H,Re,Ue,theta):
    Retheta=Re*Ue*theta;
    dndRet=0.01*((2.4*H-3.7+2.5*np.tanh(1.5*H-4.65))**2+0.25)**0.5;
    Ret0=10**((1.415/(H-1)-0.489)*np.tanh((20/(H-1))-12.9)+(3.295/(H-1))+0.44);
    n=dndRet*(Retheta-Ret0);
    return n,Retheta,Ret0

def npostxc(H,Re,Ue,theta):
    Retheta=Re*Ue*theta;
    dndRet=0.01*((2.4*H-3.7+2.5*np.tanh(1.5*H-4.65))**2+0.25)**0.5;
    p=(6.54*H-14.07)/H**2;
    m=(0.058*((H-4)**2/(H-1))-0.068)*(1/(p));
    dndx=dndRet*((m+1)/2)*p*(1/theta);
    return dndx

def dthetadx(theta,Re,Ue,duds,H):
    Retheta=Re*Ue*theta;
    Cf=0.246*(10**(-0.678*H))*Retheta**(-0.268);
    dtdx=-(theta/Ue)*(2+H)*duds+0.5*Cf;
    return dtdx,H

def h1dx(H1,dtdx,theta,Ue,duds):
    dh1dx=-H1*((1/Ue)*duds+(1/theta)*dtdx)+(0.0306/theta)*(H1-3)**(-0.6169)
    return dh1dx

def hfromh1(H1):
    if H1 <= 3.32:
        H = 3;
    elif H1 < 5.3:
        H = 0.6778 + 1.1536*(H1-3.3)**(-0.326);
    else:
        H = 1.1 + 0.86*(H1-3.3)**(-0.777);
    return H

def viscous_solver(points,alpha,Re,nu,plotting):
    [cl,cp,xc,yc,dy,ds,theta,vt]=vortex_panel(points,alpha,0)  
    nlimit=7;
    U=Re*nu
    cd=0
    cdf=0
    v=vt*U
    v=(v**2)**0.5;
    for surface in range(1,3):
        if surface==1: #lower
            b=np.argwhere(v == np.min(v))
            b=b-2; # omit possible zero gradient at LE
            b=int(b)
            vAftStag=np.fliplr([v[0:b]])[0]
            dsAftStag=np.fliplr([ds[0:b]])[0]
            xAftStag=np.fliplr([xc[0:b]])[0]
            dyAftStag=np.fliplr([dy[0:b]])[0]
            thetaAftStag=np.fliplr([theta[0:b]])[0]
            endx=b;

            s=np.zeros(endx);
            duds=np.zeros(endx)
            thetaSqr=np.zeros(endx)
            n=np.zeros(endx)
            lam=np.zeros(endx)
            Retheta=np.zeros(endx)
            Ret0=np.zeros(endx)
            H=np.zeros(endx)
            H1=np.zeros(endx)
            thetaSqr=np.ones(endx)
            thetaBL=np.zeros(endx)
            Cfx=np.zeros(endx)

        else: #upper
            b=np.argwhere(v == np.min(v))
            b=b+3; # omit possible zero gradient at LE
            b=int(b)
            vAftStag=v[b-1:-1]
            dsAftStag=ds[b-1:-1]
            xAftStag=xc[b-1:-1]
            dyAftStag=dy[b-1:-1]
            thetaAftStag=theta[b-1:-1]
            endx=len(v)-b

            s=np.zeros(endx);
            duds=np.zeros(endx)
            thetaSqr=np.zeros(endx)
            n=np.zeros(endx)
            lam=np.zeros(endx)
            Retheta=np.zeros(endx)
            Ret0=np.zeros(endx)
            H=np.zeros(endx)
            H1=np.zeros(endx)
            thetaSqr=np.ones(endx)
            thetaBL=np.zeros(endx)
            Cfx=np.zeros(endx)

        for i in range(0,endx):
            if i==0:
                if surface==1:
                    du=vAftStag[i]-v[b];
                else:
                    du=vAftStag[i]-v[b-2];
                duds[i]=du/dsAftStag[i];
            else:
                du=vAftStag[i]-vAftStag[i-1];
                duds[i]=du/dsAftStag[i];
        instability=0;
        for i in range(0,endx):
            Rex=(vAftStag[i]*sum(ds[0:i+1]))/nu
            s[i]=sum(dsAftStag[0:i+1])
            if i==0:
                thetaSqr[0]=0.075/(Re*duds[i]);
            else:
                dx=dsAftStag[i];
                xm=(s[i]+s[i-1])/2;
                integral=(dx/18)*(5*vAftStag[i]**5*(xm-((3/5)**0.5)*(dx/2))+8*vAftStag[i]**5*xm+5*vAftStag[i]**5*(xm+((3/5)**0.5)*(dx/2)))
                thetaSqr[i]=thetaSqr[i-1]+(0.45/(Re*vAftStag[i]**6))*integral;
                #print(xm)
            lam[i]=Re*thetaSqr[i]*duds[i]
            H[i]=hfromlambda(lam[i]);
            H1[i]=h1fromh(H[i]);
            (n[i],Retheta[i],Ret0[i])=nprexcr(H[i],Re,vAftStag[i],thetaSqr[i]**0.5)
            Cf=0.246*(10**(-0.678*H[i]))*Retheta[i]**(-0.268)
            Cfx[i]=Cf*dsAftStag[i]*np.cos(thetaAftStag[i])
            if Retheta[i]>Ret0[i]: 
                instability=1
                xinstab=xAftStag[i]
            if instability==1:
                dndx=npostxc(H[i],Re,vAftStag[i],thetaSqr[i]**0.5);
                if i>0:
                    n[i]=n[i-1]+dndx*dsAftStag[i];
            if n[i]>nlimit:
                I=i
                if surface==1:
                    print('Lower surface transition at xc='+str(xAftStag[i]))
                else:
                    print('Upper surface transition at xc='+str(xAftStag[i]))
                break
        I=i
        sep=0   
        for i in range(I,endx-1):
            i=int(i)
            dx=dsAftStag[i];
            h=2*dx;
            
            Retheta[i]=Re*vAftStag[i]*thetaSqr[i]**0.5;
            Cf=0.246*(10**(-0.678*H[i]))*Retheta[i]**(-0.268);
            f1=-(thetaSqr[i]**0.5/vAftStag[i])*(2+H[i])*duds[i]+0.5*Cf;            
            ystar=thetaSqr[i]**0.5+dx*f1;

            Retheta[i]=Re*vAftStag[i]*ystar;
            Cf=0.246*(10**(-0.678*H[i]))*Retheta[i]**(-0.268);
            Cfx[i]=Cf*dsAftStag[i]*np.cos(thetaAftStag[i])
            f2=-(ystar/vAftStag[i+1])*(2+H[i])*duds[i+1]+0.5*Cf;
            thetaSqr[i+1]=(thetaSqr[i]**0.5+dx*((f1+f2)/2))**2;

            if H1[i]<3.3:
                J=i-np.ceil(len(v)/200)
                sepxc=xAftStag[i]
                sep=1
                if surface==1:
                    print('Warning: lower surface separation at xc=' + str(sepxc));
                else:
                    print('Warning: upper surface separation at xc=' + str(sepxc));
                break
            else:
                k1=h1dx(H1[i],f1,thetaSqr[i]**0.5,vAftStag[i],duds[i]);
                gstar=H1[i]+dx*k1;
                if gstar<3.0:
                    J=i-np.ceil(len(v)/100)
                    sepxc=xAftStag[i]
                    sep=1
                    if surface==1:
                        print('Warning: lower surface separation at xc=' + str(sepxc));
                    else:
                        print('Warning: upper surface separation at xc=' + str(sepxc));
                    break
                k2=h1dx(gstar,f2,thetaSqr[i+1]**0.5,vAftStag[i+1],duds[i+1]);
                H1[i+1]=H1[i]+dx*((k1+k2)/2);
            H[i+1]=hfromh1(H1[i+1]);
            (n[i+1],Retheta[i+1],Ret0[i+1])=nprexcr(H[i+1],Re,vAftStag[i+1],thetaSqr[i+1]**0.5);
            if Retheta[i]<Ret0[i]: 
                instability=1
            if instability==1:
                dndx=npostxc(H[i],Re,vAftStag[i],thetaSqr[i]**0.5);
                n[i]=n[i-1]+dndx*dsAftStag[i];

        thetaBL=thetaSqr**0.5
      
        if sep==1:
            J=int(J)
            for i in range(J,endx-1):
                thetaBL[i+1]=thetaBL[i]-dyAftStag[i]/2;
                H[i+1]=2.088;
        if surface==1:             
            thetaBL=np.hstack([np.ones(3)*thetaBL[0], thetaBL])
            delta=thetaBL*2 # coarse approximation to displacement thickness 
            blxLow=xc[0:b+3] + np.fliplr([delta])[0]*np.sin(theta[0:b+3]);
            blyLow=yc[0:b+3] - np.fliplr([delta])[0]*np.cos(theta[0:b+3]);
        else:
            thetaBL=np.hstack([np.ones(2)*thetaBL[0], thetaBL])
            delta=thetaBL*2 # coarse approximation to displacement thickness
            blxUp=xc[b-3:-1] + delta * np.sin(theta[b-3:-1])
            blyUp=yc[b-3:-1] - delta * np.cos(theta[b-3:-1])
        if sep==1:
            cd = cd + abs((np.sin(alpha))**2 * (xAftStag[-1]-sepxc)**2 + 0.025*np.cos(alpha)*(xAftStag[-1]-sepxc)**2)
        else:
            cd= cd + 2 * thetaBL[-1]*(vAftStag[-1]/U)**((H[-1]+5)/2);
            cdf = cdf + sum(abs(Cfx))
    cdp = cd - cdf
    xdisp=np.hstack([blxLow[0:-2], blxUp])
    ydisp=np.hstack([blyLow[0:-2], blyUp])
    bl=np.vstack([xdisp.T,ydisp.T]).T   
    [cl,cp,xc2,yc2,dy,ds,theta,v]=vortex_panel(points,alpha,0)
    if plotting==1:
        pres,  = pl.plot(xc2,-cp,'b', label='$-C_P$')
        foil, = pl.plot(xc,yc*10,'k', label='aerofoil')
        bound, = pl.plot(bl[:,0],bl[:,1]*10,'g',label='boundary layer')
        pl.legend(handles=[pres, foil, bound])
        pl.xlabel('x/c')
        pl.ylabel(r'$-C_P$, thickness $\times 10$')
    return cl,cd,cp
