#ifndef Filon_hh
#define Filon_hh

#ifdef __cplusplus
extern "C" {
#endif


void tak_sin_integral_single(unsigned x_panels,double *xVec,double *yVec,
                        double time, double &sin_single);

void tak_cos_integral_single(unsigned x_panels,double *xVec,double *yVec,
                        double time, double &cos_single);

void tak_cal_integral(double massNum, double temperature, unsigned x_panels,double *xVec,double *yVec,
                      unsigned t_length, double *timeVec, double *gamma_classic,
                      double *gamma_quantum_real, double *gamma_quantum_imag);

void tak_cal_limit(double massNum, double temperature, double *xVec, double *yVec,
                      unsigned t_length, double *timeVec, double *limit_value,
                      double *limit_value_real, double *limit_value_imag);

void tak_cal_PDF(double rho, unsigned x_panels,double *xVec,double *yVec,
                          unsigned r_length, double *rVec, double* pdfVec);

#ifdef __cplusplus
}
#endif
#endif
