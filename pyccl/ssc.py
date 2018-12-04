from . import ccllib as lib
from .core import check
from .p2d import Pk2D
import numpy as np
import collections

# Define symbolic 'None' type for arrays, to allow proper handling by swig
# wrapper
NoneArr = np.array([])

class SSCWorkspace(object) :
    def __init__(self,cosmo,fsky,cltracer1,cltracer2,cltracer3,cltracer4,ell,response,p_of_k_a_12=None,p_of_k_a_34=None) :
        # Access ccl_cosmology object
        cosmo = cosmo.cosmo
        
        if isinstance(response,Pk2D) :
            resp=response.psp
        else :
            raise ValueError("response must be a pyccl.Pk2D object")
        
        if p_of_k_a_12 is not None :
            if isinstance(p_of_k_a_12,Pk2D) :
                psp12=p_of_k_a_12.psp
            else :
                raise ValueError("p_of_k_a_12 must be either a pyccl.Pk2D object or None")
        else :
            psp12=None
    
        if p_of_k_a_34 is not None :
            if isinstance(p_of_k_a_34,Pk2D) :
                psp34=p_of_k_a_34.psp
            else :
                raise ValueError("p_of_k_a_12 must be either a pyccl.Pk2D object or None")
        else :
            psp34=None

        # Access CCL_ClTracer objects
        clt1 = cltracer1.cltracer
        clt2 = cltracer2.cltracer
        clt3 = cltracer3.cltracer
        clt4 = cltracer4.cltracer
        
        status=0
        self.wsp,status=lib.set_ssc_workspace_new(cosmo,fsky,clt1,clt2,clt3,clt4,psp12,psp34,resp,ell,status)
        check(status)
        self.has_wsp=True
    
    def __del__(self) :
        if hasattr(self,'has_wsp'):
            if self.has_wsp:
                lib.ssc_workspace_free(self.wsp)

def angular_cl_ssc_from_workspace(ssc_w,cosmo,cltracer1,cltracer2,cltracer3,cltracer4) :
    # Access ccl_cosmology object
    cosmo = cosmo.cosmo

    # Access CCL_ClTracer objects
    clt1 = cltracer1.cltracer
    clt2 = cltracer2.cltracer
    clt3 = cltracer3.cltracer
    clt4 = cltracer4.cltracer

    status = 0
    nell=ssc_w.wsp.nell
    
    ssc, status = lib.angular_cl_ssc_from_workspace_vec(cosmo,ssc_w.wsp,clt1,clt2,clt3,clt4,
                                                        nell*nell,status)
    check(status)
    return ssc.reshape([nell,nell])

def angular_cl_ssc(cosmo, fsky, cltracer1, cltracer2, cltracer3, cltracer4, ell,
                   response,p_of_k_a_12=None,p_of_k_a_34=None) :
    """Calculate the angular (cross-)power spectrum for a pair of tracers.

    Args:
        cosmo (:obj:`Cosmology`): A Cosmology object.
        fsky (float): sky fraction
        cltracer1, cltracer2, cltracer3, cltracer4 (:obj:`Tracer`): Tracer objects, of any kind.
        ell (float or array_like): Angular wavenumber(s) at which to evaluate
            the angular power spectrum.
        response (Pk2D object or None): power spectrum response function.
            If None, the non-linear matter power spectrum will be used.
        p_of_k_a_12 (Pk2D object or None): 3D Power spectrum between tracers 1 and 2 to project.
            If None, the non-linear matter power spectrum will be used.
        p_of_k_a_34 (Pk2D object or None): 3D Power spectrum between tracers 3 and 4 to project.
            If None, the non-linear matter power spectrum will be used.

    Returns:
        float or array_like: Angular (cross-)power spectrum values,
            :math:`C_\\ell`, for the pair of tracers, as a function of
            :math:`\\ell`.
    """
    # Access ccl_cosmology object
    cosmo = cosmo.cosmo

    if isinstance(response,Pk2D) :
        resp=response.psp
    else :
        raise ValueError("response must be a pyccl.Pk2D object")
        
    if p_of_k_a_12 is not None :
        if isinstance(p_of_k_a_12,Pk2D) :
            psp12=p_of_k_a_12.psp
        else :
            raise ValueError("p_of_k_a_12 must be either a pyccl.Pk2D object or None")
    else :
        psp12=None
    
    if p_of_k_a_34 is not None :
        if isinstance(p_of_k_a_34,Pk2D) :
            psp34=p_of_k_a_34.psp
        else :
            raise ValueError("p_of_k_a_12 must be either a pyccl.Pk2D object or None")
    else :
        psp34=None
        
    # Access CCL_ClTracer objects
    clt1 = cltracer1.cltracer
    clt2 = cltracer2.cltracer
    clt3 = cltracer3.cltracer
    clt4 = cltracer4.cltracer

    status = 0
    # Return Cl values, according to whether ell is an array or not
    if isinstance(ell, float) or isinstance(ell, int):
        # Use single-value function
        nell=1
        ssc, status = lib.angular_cl_ssc_vec(cosmo,fsky,clt1,clt2,clt3,clt4,
                                             psp12,psp34,resp,[ell],1,status)
    elif isinstance(ell, np.ndarray):
        # Use vectorised function
        nell=ell.size
        ssc, status = lib.angular_cl_ssc_vec(cosmo,fsky,clt1,clt2,clt3,clt4,
                                             psp12,psp34,resp,ell,nell*nell,status)
    else:
        # Use vectorised function
        nell=len(ell)
        ssc, status = lib.angular_cl_ssc_vec(cosmo,fsky,clt1,clt2,clt3,clt4,
                                             psp12,psp34,resp,ell,nell*nell,status)
    check(status)
    return ssc.reshape([nell,nell])
