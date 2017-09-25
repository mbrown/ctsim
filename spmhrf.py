# Adaptation of SPM8's haemodynamic response function (HRF) code to python.
#
# Matthew R G Brown, PhD
# Created: 2014-05-05

import numpy as np
import scipy
import scipy.special
import warnings


def SpmHrf():
    '''Haemodynamic response function. Difference of gammas plus optional temporal and dispersion derivatives.
       Based on SPM8 Matlab code. See spm_hrf.m, spm_defaults.m, and spm_Gpdf.m
    '''

def matlabFind(a):
    '''Returns indices in column major format'''
    if len(a.shape) == 1:
        return np.array([ind for (ind,el) in enumerate(a) if el])
    elif len(a.shape) == 2:
        tmp = a.T
        index_list = []
        for rowind in range(tmp.shape[0]):
            index_list += [ind for (ind,el) in enumerate(tmp[rowind]) if el]
        return np.array(index_list)
    else:
        raise Exception('Not coded for this')

def spm_hrf(RT,P=None):
    '''Documentation copied from SPM8's spm_hrf.m
     Python code adapted from Matlab code in SPM8's spm_hrf.m

     Returns a hemodynamic response function
     FORMAT [hrf,p,x] = spm_hrf(RT,[p])
     RT   - scan repeat time
     p    - parameters of the response function (two gamma functions)
    
                                                         defaults
                                                        (seconds)
       p(1) - delay of response (relative to onset)         6
       p(2) - delay of undershoot (relative to onset)      16
       p(3) - dispersion of response                        1
       p(4) - dispersion of undershoot                      1
       p(5) - ratio of response to undershoot               6
       p(6) - onset (seconds)                               0
       p(7) - length of kernel (seconds)                   32
    
     hrf  - hemodynamic response function
     p    - parameters of the response function
     x    - abscissa values at which hrf is evaluated
    __________________________________________________________________________
     Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging

     Karl Friston
     $Id: spm_hrf.m 3716 2010-02-08 13:58:09Z karl $

     Adapted to Python by Matthew R G Brown, PhD, 2014-05-05.
    '''

    #--------------------------------------------------------------------------
    RT = float(RT)
    fMRI_T = np.int(16)

    # default parameters
    #--------------------------------------------------------------------------
    p = np.array([6,16,1,1,6,0,32],dtype=np.float)
    if P != None:
        p[0:len(P)] = P

    # modelled hemodynamic response function - {mixture of Gammas}
    #--------------------------------------------------------------------------
    dt  = RT/fMRI_T
    u   = np.arange(0, np.floor(p[6]/dt)+1) - p[5]/dt
    hrf = spm_Gpdf(u,p[0]/p[2],dt/p[2]) - spm_Gpdf(u,p[1]/p[3],dt/p[3])/p[4]
    ind = np.array( np.linspace(0,np.floor(p[6]/RT),np.floor(p[6]/RT)+1) * fMRI_T, dtype=np.int)
    hrf = hrf[ind]
    hrf = hrf / sum(hrf)
    x   = np.arange(hrf.size) * RT
    return (hrf,p,x)

def spm_Gpdf(x,h,l):
    '''Documentation copied from SPM8's spm_Gpdf.m
     Python code adapted from Matlab code in SPM8's spm_Gpdf.m

     Probability Density Function (PDF) of Gamma distribution
     FORMAT f = spm_Gpdf(x,h,l)
    
     x - Gamma-variate   (Gamma has range [0,Inf) )
     h - Shape parameter (h>0)
     l - Scale parameter (l>0)
     f - PDF of Gamma-distribution with shape & scale parameters h & l
    __________________________________________________________________________
    
     spm_Gpdf implements the Probability Density Function of the Gamma
     distribution.
    
     Definition:
    --------------------------------------------------------------------------
     The PDF of the Gamma distribution with shape parameter h and scale l
     is defined for h>0 & l>0 and for x in [0,Inf) by: (See Evans et al.,
     Ch18, but note that this reference uses the alternative
     parameterisation of the Gamma with scale parameter c=1/l)
    
               l^h * x^(h-1) exp(-lx)
        f(x) = ----------------------
                       gamma(h)
    
     Variate relationships: (Evans et al., Ch18 & Ch8)
    --------------------------------------------------------------------------
     For natural (strictly +ve integer) shape h this is an Erlang distribution.
    
     The Standard Gamma distribution has a single parameter, the shape h.
     The scale taken as l=1.
    
     The Chi-squared distribution with v degrees of freedom is equivalent
     to the Gamma distribution with scale parameter 1/2 and shape parameter v/2.
    
     Algorithm:
    --------------------------------------------------------------------------
     Direct computation using logs to avoid roundoff errors.
    
     References:
    --------------------------------------------------------------------------
     Evans M, Hastings N, Peacock B (1993)
           "Statistical Distributions"
            2nd Ed. Wiley, New York
    
     Abramowitz M, Stegun IA, (1964)
           "Handbook of Mathematical Functions"
            US Government Printing Office
    
     Press WH, Teukolsky SA, Vetterling AT, Flannery BP (1992)
           "Numerical Recipes in C"
            Cambridge
    __________________________________________________________________________
     Copyright (C) 1993-2011 Wellcome Trust Centre for Neuroimaging

     Andrew Holmes
     $Id: spm_Gpdf.m 4182 2011-02-01 12:29:09Z guillaume $

     Adapted to Python by Matthew R G Brown, PhD, 2014-05-05.
    '''

    if type(x) in [np.array,list,tuple]:
        x = np.array((x,))
    else:
        x = np.asarray(x)
    if type(h) not in [np.array,list,tuple]:
        h = np.array((h,))
    else:
        h = np.asarray(h)
    if type(l) not in [np.array,list,tuple]:
        l = np.array((l,))
    else:
        l = np.asarray(l)

    #-Format arguments, note & check sizes
    #--------------------------------------------------------------------------

    ad = np.array([len(x.shape),len(h.shape),len(l.shape)])
    rd = max(ad)
    az = np.array([np.concatenate((np.array(x.shape),np.ones(rd-ad[0]))), \
                   np.concatenate((np.array(h.shape),np.ones(rd-ad[1]))), \
                   np.concatenate((np.array(l.shape),np.ones(rd-ad[2]))) ])
    rs = np.max(az,axis=0)
    xa = np.prod(az,1) > 1
    if sum(xa)>1 and np.any(np.any(np.diff(az[xa,:],axis=0),axis=0)):
        raise Exception('Non-scalar args must match in size.')

    #-Computation
    #--------------------------------------------------------------------------
    #-Initialise result to zeros
    f = np.zeros(rs)

    #-Only defined for strictly positive h & l. Return NaN if undefined.
    md = np.logical_and(np.ones(x.shape), np.logical_and(h>0, l>0))
    if np.any(np.logical_not(md)):
        f[np.logical_not(md)] = np.nan;
        warnings.warn('Returning NaN for out of range arguments')

    #-Degenerate cases at x==0: h<1 => f=Inf; h==1 => f=l; h>1 => f=0
    ml = np.logical_and(md, np.logical_and(x==0, h<1))
    f[ml] = np.inf
    ml = np.logical_and(md, np.logical_and(x==0, h==1))
    if xa[2]:
        mll = ml
    else:
        mll = 0
    f[ml] = l[mll]

    #-Compute where defined and x>0
    Q = matlabFind(np.logical_and(md, x>0));
    if len(Q) == 0:
        return f
    if xa[0]:
        Qx = Q
    else:
        Qx = 0
    if xa[1]:
        Qh = Q
    else:
        Qh = 0
    if xa[2]:
        Ql = Q
    else:
        Ql = 0

    #-Compute
    f[Q] = np.exp( (h[Qh]-1) * np.log(x[Qx]) + h[Qh] * np.log(l[Ql]) - l[Ql] * x[Qx] - scipy.special.gammaln(h[Qh]) )

    return f
