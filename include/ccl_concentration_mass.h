/** @file */

#ifndef __CCL_CONCENTRATIONMASS_H_INCLUDED__
#define __CCL_CONCENTRATIONMASS_H_INCLUDED__

CCL_BEGIN_DECLS


/**
 * Computes concentration for a given halo mass with mass definition for a chosen concentration-mass relation
 * @param cosmo: cosmology object containing parameters
 * @param halomass: halo mass
 * @param massdef_delta_m: overdensity relative to matter density for halo size definition
 * @param cm_relation: tag of the c-m relation we want to use
 * @param a: scale factor normalised to a=1 today
 * @param status: Status flag: 0 if there are no errors, non-zero otherwise
 * @return c: halo concentration consistent with halo size definition
 */
double ccl_concentration_mass(ccl_cosmology *cosmo, double halomass, double massdef_delta_m, string cm_relation, double a, double r, int *status);


/**
 * Computes concentration for a given halo mass with mass definition for the Child+18 relation: https://arxiv.org/abs/1804.10199
 * @param halomass: halo mass
 * @param massdef_delta_m: overdensity relative to matter density for halo size definition
 * @param a: scale factor normalised to a=1 today
 * @param status: Status flag: 0 if there are no errors, non-zero otherwise
 * @return c: halo concentration consistent with halo size definition
 */
double ccl_concentration_mass_child18(double halomass, double massdef_delta_m, double a, int *status);


  CCL_END_DECLS

  #endif
