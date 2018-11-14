%module ccl_ssc

%{
  /* put additional #include here */
%}

%include "../include/ccl_ssc.h"

// Enable vectorised arguments for arrays
%apply (double* IN_ARRAY1, int DIM1) {
    (double* ell, int nell)};
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int nout, double* output)};

%inline %{

void angular_cl_ssc_vec(ccl_cosmology * cosmo,double fsky,
			CCL_ClTracer *clt1,CCL_ClTracer *clt2,
			CCL_ClTracer *clt3,CCL_ClTracer *clt4,
			ccl_p2d_t *psp12,ccl_p2d_t *psp34,ccl_p2d_t *resp,
			double* ell, int nell, int nout, double* output, int *status)
{
  //Compute C_ells
  ccl_angular_cl_ssc(cosmo,fsky,clt1,clt2,clt3,clt4,psp12,psp34,resp,nell,ell,output,status);
}

void angular_cl_ssc_from_workspace_vec(ccl_cosmology *cosmo,SSCWorkspace *w,
				       CCL_ClTracer *clt1,CCL_ClTracer *clt2,
				       CCL_ClTracer *clt3,CCL_ClTracer *clt4,
				       int nout,double *output,int *status)
{
  ccl_angular_cl_ssc_from_workspace(cosmo,w,clt1,clt2,clt3,clt4,output,status);
}
 
SSCWorkspace *set_ssc_workspace_new(ccl_cosmology *cosmo,double fsky,
				    ccl_p2d_t *psp12,ccl_p2d_t *psp34,ccl_p2d_t *resp,
				    double *ell,int nell,int *status)
{
  SSCWorkspace *wsp=ccl_ssc_workspace_new(cosmo,fsky,psp12,psp34,resp,nell,ell,status);
  return wsp;
}
 
%}
