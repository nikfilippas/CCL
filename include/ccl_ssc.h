/** @file */
#ifndef __CCL_SSC_H_INCLUDED__
#define __CCL_SSC_H_INCLUDED__

CCL_BEGIN_DECLS


typedef struct {
  double chimin;
  double chimax;
  int nchi;
  double dchi;
  double *chiarr;
  double *aarr;
  double *sbarr;
  int nell;
  double *larr;
  double *kernel;
} SSCWorkspace;

SSCWorkspace *ccl_ssc_workspace_new(ccl_cosmology *cosmo,double fsky,
				    CCL_ClTracer *clt1,CCL_ClTracer *clt2,
				    CCL_ClTracer *clt3,CCL_ClTracer *clt4,
				    ccl_p2d_t *psp1,ccl_p2d_t *psp2,
				    ccl_p2d_t *resp,int nl_out,double *l_out,
				    int *status);

void ccl_ssc_workspace_free(SSCWorkspace *w);

void ccl_angular_cl_ssc_from_workspace(ccl_cosmology *cosmo,SSCWorkspace *w,
				       CCL_ClTracer *clt1,CCL_ClTracer *clt2,
				       CCL_ClTracer *clt3,CCL_ClTracer *clt4,
				       double *ssc_out,int *status);

/**
 * Computes Limber approximation to the super-sample covariance matrix (SSC)
 * of the angular power spectrum between two power spectra C_12 and C_34, where
 * C_ij is the cross-spectrum between tracers i and j.
 * @param cosmo Cosmological parameters
 * @param clt1 a Cltracer
 * @param clt2 a Cltracer
 * @param clt3 a Cltracer
 * @param clt4 a Cltracer
 * @param psp12 the 2D power spectrum structure for tracers 1 and 2
 * @param psp34 the 2D power spectrum structure for tracers 3 and 4
 * @param resp a 2D power spectrum structure containing the response function
 * @param nl_out number of ells in which we compute the power spectra
 * @param l_out an array of ell values
 * @param ssc_out an array of size nl_out*nl_out that will contain the super-sample covariance matrix.
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return void
 */
void ccl_angular_cl_ssc(ccl_cosmology *cosmo,double fsky,
			CCL_ClTracer *clt1,CCL_ClTracer *clt2,
			CCL_ClTracer *clt3,CCL_ClTracer *clt4,
			ccl_p2d_t *psp12,ccl_p2d_t *psp34,ccl_p2d_t *resp,
			int nl_out,double*l_out,double *ssc_out,int *status);

CCL_END_DECLS


#endif
