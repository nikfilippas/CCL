#pragma once
#include "ccl_core.h"


int ccl_cosmology_serialize(
    ccl_cosmology * cosmo,
    int (*serialize_double)(double, const char *, const char *, void*),
    int (*serialize_int)(int, const char *, const char *, void*),
    int (*serialize_double_array)(int, double*, const char *, const char *, void*),
    void * context
);


int
ccl_cosmology_serialize_text_files(ccl_cosmology * cosmo, const char * dirname);

// int ccl_cosmology_deserialize(
//     ccl_cosmology * cosmo,
//     int (*deserialize_double)(double*, int*, void*),
//     int (*deserialize_int)(int*, int*, void*),
//     int (*deserialize_double_array)(int *, double**, int*, void*),
// );



// // int ccl_cosmology_serialize_directory( ccl_cosmology * cosmo, const char * dirname);
