#include "NumpyHist1D.hh"
#include "NumpyHist2D.hh"

int main()
{
  NumpyHist1D hist1(10,0.,10.);
  hist1.fill(0.1);
  hist1.fill(1.1);
  hist1.fill(2.1);
  hist1.fill(3.1);
  hist1.fill(4.1);
  hist1.fill(5.1);
  hist1.fill(6.1);
  hist1.fill(7.1);
  hist1.fill(8.1);
  hist1.fill(9.1);
  hist1.save("hist1d.npy");

  NumpyHist2D hist2(2, 0., 10.,
                    5, 0., 10.);
  hist2.fill( 2, 4.1); //0, 2
  hist2.fill( 6, 6.1); //1, 3

  hist2.save("hist2d.npy");
  return 0;
}
