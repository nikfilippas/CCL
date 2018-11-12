#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_bessel.h>

#include "ccl.h"

typedef struct {
  ccl_cosmology *cosmo;
  double a;
  double sizeR;
  int *status;
} IntSigmabPar;

static double window_sigma_b(double x)
{
  if(x<0.001)
    return 1-x*x/8.;
  else
    return 2*gsl_sf_bessel_Jn(1,x)/x;
}
  
static double sigma_b_integrand(double lk,void *params)
{
  IntSigmabPar *p=(IntSigmabPar *)params;
  double k=exp(lk);
  double pk=ccl_linear_matter_power(p->cosmo,k,p->a,p->status);
  double x=k*p->sizeR;
  double w=k*window_sigma_b(x);
  return pk*w*w;
}

static double sigma_b_integral(ccl_cosmology *cosmo,double a,double fsky,int *status)
{
  double theta=acos(1-2*fsky);
  double chi=ccl_comoving_radial_distance(cosmo,a,status);
  double sizeR=chi*theta;

  IntSigmabPar ipar;
  ipar.cosmo=cosmo;
  ipar.a=a;
  ipar.sizeR=sizeR;
  ipar.status=status;

  int gslstatus;
  double result=0,eresult;
  gsl_function F;
  F.function=&sigma_b_integrand;
  F.params=&ipar;

  gsl_integration_workspace *w=gsl_integration_workspace_alloc(ccl_gsl->N_ITERATION);
  gslstatus=gsl_integration_qag(&F,log(ccl_splines->K_MIN),log(ccl_splines->K_MAX),
				0,ccl_gsl->INTEGRATION_SIGMAR_EPSREL,ccl_gsl->N_ITERATION,
				GSL_INTEG_GAUSS15,w,&result,&eresult);
  if(gslstatus != GSL_SUCCESS) {
    ccl_raise_gsl_warning(gslstatus,"ccl_power.c: ccl_sigma_b():");
    *status|=gslstatus;
  }
  gsl_integration_workspace_free(w);

  return result/(2*M_PI);    
}

typedef struct {
  ccl_cosmology *cosmo;
  CCL_ClTracer *clt1;
  CCL_ClTracer *clt2;
  CCL_ClTracer *clt3;
  CCL_ClTracer *clt4;
  ccl_p2d_t *psp12;
  ccl_p2d_t *psp34;
  ccl_p2d_t *resp;
  double lmean;
  double lfac12; // log((l12+0.5)/(lmean+0.5))
  double lfac34; // log((l34+0.5)/(lmean+0.5))
  SplPar *sp_sigmab;
  int *status;
} IntSSCPar;

static double sigma_b(IntSSCPar *p,double a)
{
  //  fprintf(stderr,"Sigma_b Not implemented\n");
  //  exit(1);
  return ccl_spline_eval(a,p->sp_sigmab);
}

static double transfer_nc(double l,double k,double chi,double a,
			  ccl_cosmology *cosmo,CCL_ClTracer *clt,int *status)
{
  double ret=0;
  if(chi<=clt->chimax) {
    double z=1./a-1;
    double pz=ccl_spline_eval(z,clt->spl_nz);
    double bz=ccl_spline_eval(z,clt->spl_bz);
    double h=cosmo->params.h*ccl_h_over_h0(cosmo,a,status)/CLIGHT_HMPC;
    double f_all=pz*bz*h;
    if(clt->has_rsd) {
      fprintf(stderr,"Not implemented\n");
      exit(1);
    }
    if(clt->has_magnification) {
      fprintf(stderr,"Not implemented\n");
      exit(1);
    }
    ret=f_all;
  }

  return ret;
}

static double transfer_wl(double l,double k,double chi,double a,
			  ccl_cosmology *cosmo,CCL_ClTracer *clt,int *status)
{
  double ret=0;
  if(chi<=clt->chimax) {
    double f_all;
    double wL=ccl_spline_eval(chi,clt->spl_wL);

    if(wL<=0)
      f_all=0;
    else
      f_all=clt->prefac_lensing*wL/(a*chi);

    if(clt->has_intrinsic_alignment) {
      fprintf(stderr,"Not implemented\n");
      exit(1);
    }
    ret=f_all;
  }
  
  return sqrt((l+2.)*(l+1.)*l*(l-1.))*ret/(k*k);
}

static double transfer_cmblens(double l,double k,double chi,double a,
			       ccl_cosmology *cosmo,CCL_ClTracer *clt,int *status)
{
  if(chi>=clt->chi_source)
    return 0;

  if(chi<=clt->chimax) {
    double w=1-chi/clt->chi_source;
    return clt->prefac_lensing*l*(l+1.)*w/(a*chi*k*k);
  }
  return 0;
}

static double transfer_wrap(double l,double k,double chi,double a,
			    ccl_cosmology *cosmo,CCL_ClTracer *clt,int *status)
{
  double transfer_out=0;

  if(clt->tracer_type==ccl_number_counts_tracer)
    transfer_out=transfer_nc(l,k,chi,a,cosmo,clt,status);
  else if(clt->tracer_type==ccl_weak_lensing_tracer)
    transfer_out=transfer_wl(l,k,chi,a,cosmo,clt,status);
  else if(clt->tracer_type==ccl_cmb_lensing_tracer)
    transfer_out=transfer_cmblens(l,k,chi,a,cosmo,clt,status);
  else
    transfer_out=-1;
  return transfer_out;
}

static double cl_ssc_integrand(double lk,void *params)
{
  double d1,d2,d3,d4;
  IntSSCPar *p=(IntSSCPar *)params;
  double k=exp(lk);
  double chi=(p->lmean+0.5)/k;
  double a=ccl_scale_factor_of_chi(p->cosmo,chi,p->status);

  d1=transfer_wrap(p->lmean,k,chi,a,p->cosmo,p->clt1,p->status);
  if(d1==0)
    return 0;
  d2=transfer_wrap(p->lmean,k,chi,a,p->cosmo,p->clt2,p->status);
  if(d2==0)
    return 0;
  d3=transfer_wrap(p->lmean,k,chi,a,p->cosmo,p->clt3,p->status);
  if(d3==0)
    return 0;
  d4=transfer_wrap(p->lmean,k,chi,a,p->cosmo,p->clt4,p->status);
  if(d4==0)
    return 0;

  double lk12=p->lfac12+lk;
  double lk34=p->lfac34+lk;
  double p12=ccl_p2d_t_eval(p->psp12,lk12,a,p->cosmo,p->status);
  double p34=ccl_p2d_t_eval(p->psp34,lk34,a,p->cosmo,p->status);
  double r12=ccl_p2d_t_eval(p->resp,lk12,a,p->cosmo,p->status);
  double r34=ccl_p2d_t_eval(p->resp,lk34,a,p->cosmo,p->status);
  double sigmab=sigma_b(p,a);

  return d1*d2*d3*d4*r12*p12*r34*p34*sigmab/(chi*chi*chi);
}

//Figure out k intervals where the Limber kernel has support
//clt1 -> tracer #1
//clt2 -> tracer #2
//l    -> angular multipole
//lkmin, lkmax -> log of the range of scales where the transfer functions have support
static void get_k_interval(ccl_cosmology *cosmo,
			   CCL_ClTracer *clt1,CCL_ClTracer *clt2,double l,
			   double *lkmin,double *lkmax)
{
  double chimin,chimax;
  int cut_low_1=0,cut_low_2=0;

  //Define a minimum distance only if no lensing is needed
  if((clt1->tracer_type==ccl_number_counts_tracer) && (clt1->has_magnification==0)) cut_low_1=1;
  if((clt2->tracer_type==ccl_number_counts_tracer) && (clt2->has_magnification==0)) cut_low_2=1;

  if(cut_low_1) {
    if(cut_low_2) {
      chimin=fmax(clt1->chimin,clt2->chimin);
      chimax=fmin(clt1->chimax,clt2->chimax);
    }
    else {
      chimin=clt1->chimin;
      chimax=clt1->chimax;
    }
  }
  else if(cut_low_2) {
    chimin=clt2->chimin;
    chimax=clt2->chimax;
    }
  else {
    chimin=0.5*(l+0.5)/ccl_splines->K_MAX;
    chimax=2*(l+0.5)/ccl_splines->K_MIN;
  }
  
  if(chimin<=0)
    chimin=0.5*(l+0.5)/ccl_splines->K_MAX;
  
  *lkmax=log(fmin( ccl_splines->K_MAX  ,2  *(l+0.5)/chimin));
  *lkmin=log(fmax( ccl_splines->K_MIN  ,0.5*(l+0.5)/chimax));
}

static double ccl_angular_cl_ssc_native(ccl_cosmology *cosmo,SplPar *sp_sigmab,
					CCL_ClTracer *clt1,CCL_ClTracer *clt2,
					CCL_ClTracer *clt3,CCL_ClTracer *clt4,
					ccl_p2d_t *psp12,ccl_p2d_t *psp34,ccl_p2d_t *resp,
					double l12,double l34,int *status)
{
  IntSSCPar ipar;
  int clastatus=0;
  double lmean=0.5*(l12+l34);
  int gslstatus;
  double result=0,eresult;
  gsl_function F;
  gsl_integration_workspace *w=gsl_integration_workspace_alloc(ccl_gsl->N_ITERATION);

  ipar.sp_sigmab=sp_sigmab;
  ipar.lmean=lmean;
  ipar.lfac12=log((l12+0.5)/(lmean+0.5));
  ipar.lfac34=log((l34+0.5)/(lmean+0.5));
  ipar.cosmo=cosmo;
  ipar.clt1=clt1;
  ipar.clt2=clt2;
  ipar.clt3=clt3;
  ipar.clt4=clt4;
  ipar.psp12=psp12;
  ipar.psp34=psp34;
  ipar.resp=resp;
  ipar.status=&clastatus;
  F.function=&cl_ssc_integrand;
  F.params=&ipar;

  double lkmin12,lkmax12,lkmin34,lkmax34,lkmin,lkmax;
  get_k_interval(cosmo,clt1,clt2,l12,&lkmin12,&lkmax12);
  get_k_interval(cosmo,clt3,clt4,l34,&lkmin34,&lkmax34);
  lkmin=fmin(lkmin12,lkmin34);
  lkmax=fmax(lkmax12,lkmax34);

  gslstatus=gsl_integration_qag(&F,lkmin,lkmax,0,ccl_gsl->INTEGRATION_LIMBER_EPSREL,
				ccl_gsl->N_ITERATION,ccl_gsl->INTEGRATION_LIMBER_GAUSS_KRONROD_POINTS,
				w,&result,&eresult);
  gsl_integration_workspace_free(w);

  // Test if a round-off error occured in the evaluation of the integral
  // If so, try another integration function, more robust but potentially slower
  /*
  if(gslstatus == GSL_EROUND) {
    ccl_raise_gsl_warning(gslstatus, "ccl_cls.c: ccl_angular_cl_ssc_native(): Default GSL integration failure, attempting backup method.");
    gsl_integration_cquad_workspace *w_cquad= gsl_integration_cquad_workspace_alloc(ccl_gsl->N_ITERATION);
    size_t nevals=0;
    gslstatus=gsl_integration_cquad(&F, lkmin, lkmax, 0,
				    ccl_gsl->INTEGRATION_LIMBER_EPSREL,
				    w_cquad, &result, &eresult, &nevals);
    gsl_integration_cquad_workspace_free(w_cquad);
  }
  if(gslstatus!=GSL_SUCCESS || *ipar.status) {
    ccl_raise_gsl_warning(gslstatus, "ccl_cls.c: ccl_angular_cl_ssc_native():");
    // If an error status was already set, don't overwrite it.
    if(*status == 0){
      *status=0;//CCL_ERROR_INTEG;
        ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: ccl_angular_cl_ssc_native(): error integrating over k\n");
    }
    return result;
    }*/
  ccl_check_status(cosmo,status);

  return result;
}
    
void ccl_angular_cl_ssc(ccl_cosmology *cosmo,double fsky,
			CCL_ClTracer *clt1,CCL_ClTracer *clt2,
			CCL_ClTracer *clt3,CCL_ClTracer *clt4,
			ccl_p2d_t *psp12,ccl_p2d_t *psp34,ccl_p2d_t *resp,
			int nl_out,double *l_out,double *ssc_out,int *status)
{
  int i12,i34;
  ccl_p2d_t *psp12_use,*psp34_use;

  if(*status==0) {
    if(psp12==NULL) {
      if (!cosmo->computed_power) ccl_cosmology_compute_power(cosmo, status);
      psp12_use=cosmo->data.p_nl;
    }
    else
      psp12_use=psp12;
  }

  if(*status==0) {
    if(psp12==NULL) {
      if (!cosmo->computed_power) ccl_cosmology_compute_power(cosmo, status);
      psp34_use=cosmo->data.p_nl;
    }
    else
      psp34_use=psp34;
  }
  
  if(*status==0) {
    if(resp==NULL) {
      fprintf(stderr,"Must provide response for SSC\n");
      exit(1);
    }
  }

  int na;
  double *a_sb,*s_sb;
  SplPar *sp_sigmab;
  if(*status==0) {
    //Compute sigma_b
    na=100;
    a_sb=ccl_linear_spacing(0.01,1.,na);
    if (a_sb==NULL ||
	(fabs(a_sb[0]-0.01)>1e-5) ||
	(fabs(a_sb[na-1]-1.)>1e-5) ||
	(a_sb[na-1]>1.0)) {
      // old:    cosmo->status = CCL_ERROR_LINSPACE;
      *status = CCL_ERROR_LINSPACE;
      ccl_cosmology_set_status_message(cosmo, "ccl_ssc.c: ccl_angular_cl_ssc(): Error creating first logarithmic and then linear spacing in a\n");
    }
  }

  if(*status==0) {
    s_sb=malloc(sizeof(double)*na);
    if(s_sb==NULL) { 
      *status=CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo, "ccl_ssc.c: ccl_angular_cl_ssc(): ran out of memory\n");
    }
  }
 
  if(*status==0) {
    for(int i=0;i<na;i++)
      s_sb[i]=sigma_b_integral(cosmo,a_sb[i],fsky,status);
    sp_sigmab=ccl_spline_init(na,a_sb,s_sb,0,0);
  }

  if(*status==0) {
    for(i12=0;i12<nl_out;i12++) {
      for(i34=0;i34<nl_out;i34++) {
	ssc_out[i34+nl_out*i12]=ccl_angular_cl_ssc_native(cosmo,sp_sigmab,
							  clt1,clt2,clt3,clt4,
							  psp12_use,psp34_use,resp,
							  l_out[i12],l_out[i34],status);
      }
    }
  }

  ccl_check_status(cosmo,status);
  free(a_sb); free(s_sb); ccl_spline_free(sp_sigmab);
}
