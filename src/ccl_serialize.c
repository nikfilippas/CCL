#include "ccl_serialize.h"
#include "stdio.h"
#include "string.h"


 
struct text_file_context {
    const char * dirname;
    FILE * params;
};

static 
int serialize_double_text_file(double value, const char * category, const char * name, void* context){
    struct text_file_context * tf_context = (struct text_file_context*) context;
    fprintf(tf_context->params, "%s = %le\n", name, value);
    return 0;
}

static 
int serialize_int_text_file(int value, const char * category, const char * name, void* context){
    struct text_file_context * tf_context = (struct text_file_context*) context;
    fprintf(tf_context->params, "%s = %d\n", name, value);
    return 0;
}

static 
int serialize_array_text_file(int len, double *value, const char * category, const char * name, void* context){
    struct text_file_context * tf_context = (struct text_file_context*) context;
    char filename[512];
    snprintf(filename, 512, "%s/%s_%s.txt", tf_context->dirname, category, name);

    FILE * f = fopen(filename, "w");
    for (int i=0; i<len; i++){
        fprintf(f, "%le\n", value[i]);
    }
    fclose(f);
    return 0;
    
}


int
ccl_cosmology_serialize_text_files(ccl_cosmology * cosmo, const char * dirname){


    // Make the parameter file
    // Sure there is a better way of doing this
    char params_filename[512];
    snprintf(params_filename, 512, "%s/%s", dirname, "parameters.txt");

    struct text_file_context context;
    context.dirname = dirname;
    context.params = fopen(params_filename, "w");

    int status = ccl_cosmology_serialize(cosmo, serialize_double_text_file, serialize_int_text_file, serialize_array_text_file, &context);

    return status;
}


static 
int serialize_parameters(ccl_parameters * params, 
    int (*serialize_double)(double, const char *, const char *, void*),
    int (*serialize_int)(int, const char *, const char *, void*),
    int (*serialize_double_array)(int, double*, const char *, const char *, void*),
    void* context)
{

#define SERIALIZE_DOUBLE(x) serialize_double(params->x, "parameters", #x, context)
#define SERIALIZE_INT(x) serialize_int(params->x, "parameters", #x, context)
#define SERIALIZE_ARRAY(n,x) serialize_double_array(n, params->x, "parameters", #x, context)

    int status = 0;
    // Densities: CDM, baryons, total matter, neutrinos, curvature
    status |= SERIALIZE_DOUBLE(Omega_c); /**< Density of CDM relative to the critical density*/
    status |= SERIALIZE_DOUBLE(Omega_b); /**< Density of baryons relative to the critical density*/
    status |= SERIALIZE_DOUBLE(Omega_m); /**< Density of all matter relative to the critical density*/
    status |= SERIALIZE_DOUBLE(Omega_k); /**< Density of curvature relative to the critical density*/
    status |= SERIALIZE_DOUBLE(sqrtk); /**< Square root of the magnitude of curvature, k */ //TODO check
    status |= SERIALIZE_INT(k_sign); /**<Sign of the curvature k */

    status |= SERIALIZE_DOUBLE(w0);
    status |= SERIALIZE_DOUBLE(wa);

    // Hubble parameters
    status |= SERIALIZE_DOUBLE(H0);
    status |= SERIALIZE_DOUBLE(h);


    status |= SERIALIZE_DOUBLE(N_nu_mass); // Number of different species of massive neutrinos
    status |= SERIALIZE_DOUBLE(N_nu_rel);  // Neff massless
    status |= SERIALIZE_DOUBLE(mnu);  // total mass of massive neutrinos
    status |= SERIALIZE_DOUBLE(Omega_n_mass); // Omega_nu for MASSIVE neutrinos 
    status |= SERIALIZE_DOUBLE(Omega_n_rel); // Omega_nu for MASSLESS neutrinos

    status |= SERIALIZE_DOUBLE(A_s);
    status |= SERIALIZE_DOUBLE(n_s);

    status |= SERIALIZE_DOUBLE(Omega_g);
    status |= SERIALIZE_DOUBLE(T_CMB);

    status |= SERIALIZE_DOUBLE(sigma_8);
    status |= SERIALIZE_DOUBLE(Omega_l);
    status |= SERIALIZE_DOUBLE(z_star);

    status |= SERIALIZE_INT(has_mgrowth);

    if (params->has_mgrowth){
        status |= SERIALIZE_ARRAY(params->nz_mgrowth, z_mgrowth);
        status |= SERIALIZE_ARRAY(params->nz_mgrowth, df_mgrowth);
    }

    return status;

#undef SERIALIZE_DOUBLE
#undef SERIALIZE_INT
#undef SERIALIZE_ARRAY

}



#define SERIALIZE_SPLINE(g,cat) \
    serialize_int(cosmo->data.g->size, cat, #g ".size", context); \
    serialize_double_array(cosmo->data.g->size, cosmo->data.g->x, cat, #g ".x", context); \
    serialize_double_array(cosmo->data.g->size, cosmo->data.g->y, cat, #g ".y", context) \




int ccl_cosmology_serialize(
    ccl_cosmology * cosmo,
    int (*serialize_double)(double, const char *, const char *, void*),
    int (*serialize_int)(int, const char *, const char *, void*),
    int (*serialize_double_array)(int, double*, const char *, const char *, void*),
    void * context
)
{
    int status = 0 ;
    status |= serialize_parameters(&(cosmo->params),  serialize_double, serialize_int, serialize_double_array, context);

    if (cosmo->computed_distances){
        SERIALIZE_SPLINE(chi, "distances");
        SERIALIZE_SPLINE(E, "distances");
        SERIALIZE_SPLINE(achi, "distances");
    }

    if (cosmo->computed_growth){
        serialize_double(cosmo->data.growth0, "growth", "growth0", context);
        SERIALIZE_SPLINE(chi, "growth");
        SERIALIZE_SPLINE(growth, "growth");
        SERIALIZE_SPLINE(fgrowth, "growth");
    }

    if (cosmo->computed_sigma){
        SERIALIZE_SPLINE(logsigma, "sigma");
        SERIALIZE_SPLINE(dlnsigma_dlogm, "sigma");
    }


    return status;
}

