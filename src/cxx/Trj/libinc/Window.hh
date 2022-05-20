#ifndef Filon_hh
#define Filon_hh
#ifdef __cplusplus
extern "C" {
#endif

void kaiser(double beta, unsigned M, double *window);
void gauss(double alpha, unsigned M, double *window);
void blackmann(unsigned M, double *window);
void hanning(unsigned M, double *window);
void hft(unsigned M, double *window, unsigned order=3);

#ifdef __cplusplus
}
#endif
#endif
