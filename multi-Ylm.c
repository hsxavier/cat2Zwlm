// To compile: gcc -fPIC -fopenmp -shared multi-Ylm.c -o multi-Ylm.so -lm -lgsl -lgslcblas

#include <stddef.h>
#include <gsl/gsl_sf_legendre.h>
#include <math.h>
#include <omp.h>                 // For OpenMP functions, not pragmas.
#include "selected-utils.c"


/*** This function computes the value of all multipoles m>0 up to 'lmax' for all 'Ncoords' 
     coordinates (theta,phi) in the vectors 'theta_void' and 'phi_void'. The real part 
     of the results are placed in 'Re_Ylm_void' while the imaginary part is placed in 
     'Im_Ylm_void', which are one-dimensional arrays. For coordinates i, the result 
     is arranjed in the order (l,m,i): (0,0,0), (0,0,1), (0,0,2), ..., (1,0,0), (1,0,1), 
     (1,0,2), ..., (1,1,0), (1,1,1), (1,1,2), ..., (2,0,0), (2,0,1), ...
 ***/
void multiYlm(int lmax, const void *theta_void, const void * phi_void, int Ncoords, void * Re_Ylm_void, void * Im_Ylm_void) {
  // Declare internal variables:
  double **Plm;
  int NPlms, Nthreads, thread, i, l, m, j;
  // Recast input and output into their true types: 
  const double * theta = (double *) theta_void;
  const double * phi   = (double *)   phi_void;
  double * Re_Ylm = (double *) Re_Ylm_void;
  double * Im_Ylm = (double *) Im_Ylm_void;
  
  // Allocate memory:
  Nthreads = omp_get_max_threads();
  NPlms    = gsl_sf_legendre_array_n(lmax);
  Plm      = dmatrix(0,Nthreads-1, 0,NPlms-1);
  
  // Loop over (theta,phi):
#pragma omp parallel for private(thread,l,m,j)
  for(i=0; i<Ncoords; i++) {
    thread = omp_get_thread_num();
    // Compute associated Legendre polynomials up to lmax:
    gsl_sf_legendre_array_e(GSL_SF_LEGENDRE_SPHARM, lmax, cos(theta[i]), -1, Plm[thread]);
    // Apply phi phase to each multipole and save result:
    for (l=0; l<=lmax; l++) 
      for (m=0; m<=l; m++) {
	j = l*(l+1)/2 + m;
	Re_Ylm[j*Ncoords + i] = Plm[thread][j]*cos(m*phi[i]);
	Im_Ylm[j*Ncoords + i] = Plm[thread][j]*sin(m*phi[i]);
      }
  }
  
  // Free memory:
  free_dmatrix(Plm, 0,Nthreads-1, 0,NPlms-1);
}
