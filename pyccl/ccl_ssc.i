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

%}
