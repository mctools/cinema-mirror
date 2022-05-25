#include <cmath>
#include <cstring> //memcpy
#include <numeric> //std::inner_product

#include <fstream>
#include "Utils.hh"
#include <thread>
#include <future>
#include <chrono>
#include <algorithm>
#include "PTProgressMonitor.hh"
#include "Window.hh"
#include "Trajectory.hh"

#include "omp.h"
#pragma omp declare reduction( + : std::vector< double > : \
        std::transform(omp_in.begin( ),  omp_in.end( ), \
                       omp_out.begin( ), omp_out.begin( ), \
                       std::plus< double>( )) ) \
                       initializer (omp_priv(std::vector<double>(omp_orig.size(),0)))

Trajectory::Trajectory(std::string fileName)
:m_fileName(fileName)
{
}

Trajectory::~Trajectory()
{

}

size_t Trajectory::atomVdosFFTSize() const
{
  return m_nFrame*2;
  // return nextGoodFFTNumber(m_nFrame*2);
}

Data<double> Trajectory::getOmegaVector() const
{
  Data<double> omega = unweightedFrequency();
  double freScale =getDeltaT()/(2*M_PI);
  omega.vec = omega.vec *(1./freScale);
  return omega;
}


Data<double> Trajectory::unweightedFrequency() const
{
  // fixme: the size of fftSize must be even
  // because of the limiation in ths function
  size_t fftSize= atomVdosFFTSize();
  if(fftSize%2==1)
      throw std::runtime_error ("atomVdosFFTSize must be even");
  Data<double> fre;
  fre.vec = fftfreq(fftSize);
  fftshift(fre.vec);
  fre.vec.push_back(-fre.vec[0]);
  fre.vec.erase (fre.vec.begin(),fre.vec.begin()+fftSize/2);
  fre.dim.resize(1);
  fre.dim[0]=fftSize/2+1;
  return fre; //as defined in vDosSqw and structFactSqw
}

void Trajectory::coherentDynamicsFactor(unsigned qSize, Data<double> &sqw, std::vector<double> &scat_length) const
{
  if(scat_length.empty())
    throw std::runtime_error ("scattering length vector empty ");
  if(scat_length.size() != m_nAtomPerMole)
    throw std::runtime_error ("The element number in the scattering length vector should match atom number per mole ");

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  //set the correct size
  //As suggested in G.R. Kneller Comp. Phys. Comm., 1995,
  //input is padded with zero to double the size
  const size_t frame_used = m_nFrame; //!!! this is the window size
  const size_t fftSize= frame_used*2;
  const size_t omegaSize=fftSize/2+1;

  std::vector<double> Q(qSize);
  if(!m_qResolution)
     throw std::runtime_error ("m_qResolution is zero ");

  for(unsigned i=0;i<qSize;i++)
  {
    Q[i]=(i+1)*m_qResolution;
  }

  sqw.vec.resize(0);
  sqw.vec.resize(Q.size()*omegaSize, 0.);
  std::vector<double>& sqwVec = sqw.vec;

  unsigned numKpoint=3;
  size_t kdatasize=m_nAtom*omegaSize;

  std::vector<double> rho_qt_real(kdatasize*numKpoint, 0.);
  std::vector<double> rho_qt_imag(kdatasize*numKpoint, 0.);
  double add(0.), mul(0.), fma(0.);

  Prompt::ProgressMonitor* moni = new Prompt::ProgressMonitor("coherentDynamicsFactor", m_nAtom*Q.size(), 0.05);

  std::vector<double> window(frame_used);
  kaiser(20, window.size(), window.data());

  for(unsigned Qidx=0;Qidx<Q.size();Qidx++)
  {
    printf("Q index %d\n", Qidx);

    //fill rho_qt for the further matrix operation
    std::fill(rho_qt_real.data(), rho_qt_real.data() + rho_qt_real.size(), 0.);
    std::fill(rho_qt_imag.data(), rho_qt_imag.data() + rho_qt_imag.size(), 0.);
    double *rho_qt_real_data=rho_qt_real.data();
    double *rho_qt_imag_data=rho_qt_imag.data();
    std::vector<double> sqw_fixQ(omegaSize, 0.);

    #pragma omp parallel default(none) shared(rho_qt_real_data, rho_qt_imag_data, moni, window) firstprivate(frame_used, scat_length, kdatasize, Q, Qidx, fftSize, omegaSize) reduction (+: add, mul, fma, sqw_fixQ)
    {

      double tadd(0.), tmul(0.), tfma(0.);

      unsigned numThread = omp_get_num_threads();
      unsigned threadID = omp_get_thread_num();
      if(!threadID && Qidx==0)
          printf(" coherentDynamicsFactor uses %d threads\n", numThread);

      // create pointers
      fftw_complex *fftwComplexBuffer = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*fftSize);
      fftw_complex *fftwComplexBuffer_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*fftSize);

      fftw_plan    fftPlan_c2c;
      auto complexVecAsDouble = reinterpret_cast< double* > (fftwComplexBuffer);
      std::fill(complexVecAsDouble, complexVecAsDouble+fftSize*2, 0.);

      // create FFTW plan
      #pragma omp critical
      {
       fftPlan_c2c = fftw_plan_dft_1d(fftSize, fftwComplexBuffer, fftwComplexBuffer_out, FFTW_FORWARD, FFTW_ESTIMATE |  FFTW_PATIENT);
      }


      //memory size of sqw grow linearly in Q size, so critical is used instead of reduction on sqw
      #pragma omp for
      for(unsigned atomID=0;atomID<m_nAtom;atomID++)
      {
        //load atom trajectory
        const double* const trj = readAtomTrj(atomID);
        double atom_scl = scat_length[atomID%m_nAtomPerMole];

        for(unsigned dim=0;dim<3;dim++)
        {
          for(unsigned k=0;k<frame_used;k++)
          {
            double x = Q[Qidx]*trj[k*3+dim];
            double cx(cos(x)), sx(sin(x));
            fftwComplexBuffer[k][0] = cx*window[k];
            fftwComplexBuffer[k][1] = sx*window[k];
          }

          fftw_execute(fftPlan_c2c);
          fftw_flops(fftPlan_c2c, &tadd, &tmul, &tfma);
          add += tadd;
          mul += tmul;
          fma += tfma;
          for(size_t i=0;i<omegaSize;++i)
          {
            rho_qt_real_data[kdatasize*dim + i*m_nAtom + atomID] = atom_scl * fftwComplexBuffer_out[i][0];
            rho_qt_imag_data[kdatasize*dim + i*m_nAtom + atomID] = atom_scl* fftwComplexBuffer_out[i][1];
          }
        }
        moni->OneTaskCompleted();
       }

      #pragma omp critical
      {
         fftw_cleanup();
         fftw_destroy_plan(fftPlan_c2c);
      }
      // clean up
      fftw_free(fftwComplexBuffer);
      fftw_free(fftwComplexBuffer_out);

      #pragma omp for
      for(size_t freidx=0;freidx<omegaSize;freidx++)
      {
        StableSum ssum;
        for(unsigned dim=0;dim<3;dim++)
        {
          const double *pt_real = rho_qt_real_data+freidx*m_nAtom+kdatasize*dim;
          const double *pt_imag = rho_qt_imag_data+freidx*m_nAtom+kdatasize*dim;

          ssum.clear();
          for(size_t idx=0; idx<m_nAtom; idx++)
            ssum.add(pt_real[idx]);

          sqw_fixQ[freidx]+=ssum.sum()*ssum.sum();

          ssum.clear();
          for(size_t idx=0; idx<m_nAtom; idx++)
            ssum.add(pt_imag[idx]);

          sqw_fixQ[freidx]+=ssum.sum()*ssum.sum();
        }
      }
    }

    //out of the parallel section
    // memcpy(sqwVec.data()+omegaSize*Qidx, sqw_fixQ.data(), omegaSize*sizeof(double) );
    for(size_t omidx=0; omidx<omegaSize; ++omidx)
    {
      sqwVec[omegaSize*Qidx+omidx] += sqw_fixQ[omidx]; //fixme saving the last value
    }

  }
  delete moni;

  double scale = numKpoint*frame_used*frame_used*m_nAtom; //sqw is calculated for an atom in each molecule, and with 3 dimensions.
  double weight = 1./scale;
  for(auto &v: sqw.vec)
    v *= weight;

  sqw.dim.resize(2);
  sqw.dim[0]=Q.size();
  sqw.dim[1]=omegaSize;
  sqw.sanityCheck();

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Number of FFT math operations: additions " << add
        << ", multiplications "<< mul
        << ", multiply-add operations " << fma << std::endl;

  std::cout << "Elapsed time "
        << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
        << "[ms]" << std::endl << std::endl;
}


void Trajectory::vDosSqw(unsigned firstAtomPos, Data<double> &vdos, Data<double> &omega, const std::vector<double> &Q,
  Data<double> &sqw) const
{
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  //set the correct size
  //As suggested in G.R. Kneller Comp. Phys. Comm., 1995,
  //input is padded with zero to double the size
  size_t fftSize= atomVdosFFTSize();
  size_t vodsSize=fftSize/2+1;
  vdos.vec.resize(0);
  vdos.vec.resize(vodsSize, 0.);

  if(!Q.empty())
  {
    sqw.vec.resize(0);
    sqw.vec.resize(Q.size()*vodsSize, 0.);
  }
  std::vector<double>& vdosVec = vdos.vec;
  std::vector<double>& sqwVec = sqw.vec;

  // for the purpose of monitoring add, mul and multiply-add operation of the fft process
  //     void fftw_flops(const fftw_plan plan,
  //                    double *add, double *mul, double *fma);
  // quote http://www.fftw.org/fftw3_doc/Using-Plans.html:
  // Given a plan, set add, mul, and fma to an exact count of the number of
  // floating-point additions, multiplications, and fused multiply-add
  // operations involved in the plan’s execution. The total number of
  // floating-point operations (flops) is add + mul + 2*fma, or add + mul + fma
  // if the hardware supports fused multiply-add instructions
  // (although the number of FMA operations is only approximate because of
  // compiler voodoo).

  std::vector<double> window(m_nFrame);
  kaiser(20, window.size(), window.data());

  double add(0.), mul(0.), fma(0.);
  #pragma omp parallel default(none) shared(window) firstprivate(Q, firstAtomPos, fftSize, vodsSize) reduction (+ : vdosVec, sqwVec, add, mul, fma)
  {
    double tadd(0.), tmul(0.), tfma(0.);

    std::vector<StableSum> stableVdos(vdosVec.size());

    Prompt::ProgressMonitor *moni(nullptr);
    unsigned numThread = omp_get_num_threads();
    unsigned threadID = omp_get_thread_num();
    unsigned threadDoesPrint = numThread-1;
    if(threadID==threadDoesPrint)
        printf(" vDosSqw uses %d threads\n", numThread);

     // create pointers
     fftw_plan       fftPlan_r2c, fftPlan_c2c;
     std::vector<double> trj;

     size_t num_timeivt=m_nFrame-1;
     size_t frame_used = num_timeivt; //!!!here is the window size
     size_t frame_step = 0.1*num_timeivt;

     // preallocate vectors for FFTW
     //sqwfftwComplexBuffer could be (fftSize/2+1) for vdos, use a greater one in case of sqw
     fftw_complex *sqwfftwComplexBuffer = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*fftSize);
     fftw_complex *sqwfftwComplexBuffer_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*fftSize);
     auto buffAsComplexVec = reinterpret_cast<std::complex<double>* > (sqwfftwComplexBuffer);


     //vdos buffer
     fftw_complex *vdosfftwComplexBuffer = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*fftSize);
     double *realBuffer  = (double *)fftw_malloc(sizeof(double)*fftSize);
     std::fill(realBuffer, realBuffer+ fftSize, 0.); //padding with zero

     // create FFTW plan
     #pragma omp critical
     {
       fftPlan_r2c = fftw_plan_dft_r2c_1d(fftSize, realBuffer, vdosfftwComplexBuffer, FFTW_ESTIMATE |  FFTW_PATIENT);
       if(!Q.empty())
         fftPlan_c2c = fftw_plan_dft_1d(fftSize, sqwfftwComplexBuffer, sqwfftwComplexBuffer_out, FFTW_FORWARD, FFTW_ESTIMATE  |  FFTW_PATIENT);
     }


     if(threadID==threadDoesPrint)
     {
       moni = new Prompt::ProgressMonitor("Task vDosSqw", m_nMolecule/numThread);
       printf("Time window size %gps, window step size %gps, total steps %d\n", frame_used*m_deltaT*1e12, frame_step*m_deltaT*1e12, int((num_timeivt-frame_used)/frame_step)+1 );
     }

     // std::vector<double> vdoswin(frame_used*2);
     // kaiser(20, vdoswin.size(), vdoswin.data());


     //memory size of sqw grow linearly in Q size, so critical is used instead of reduction on sqw
     #pragma omp for simd
     for(unsigned i=0;i<m_nMolecule;i++)
     {
       //load atom trajectory
       unsigned atomID = i*m_nAtomPerMole + firstAtomPos;
       readAtomTrj(atomID, trj);
       unwrap(trj);

       //vDos
       for(unsigned dim=0;dim<3;dim++)
       {
         for(size_t iniFrame=0; iniFrame <= num_timeivt-frame_used; iniFrame=iniFrame+frame_step)
         {
           for(size_t k=iniFrame;k<iniFrame+frame_step;++k)
           {
             realBuffer[k-iniFrame]=(trj[(k+1)*3+dim]-trj[k*3+dim]);
           }
           fftw_execute(fftPlan_r2c);
           fftw_flops(fftPlan_r2c, &tadd, &tmul, &tfma);
           add += tadd;
           mul += tmul;
           fma += tfma;

           for(unsigned j=0;j<frame_used;j++)
           {
             stableVdos[j].add(vdosfftwComplexBuffer[j][0]*vdosfftwComplexBuffer[j][0]+vdosfftwComplexBuffer[j][1]*vdosfftwComplexBuffer[j][1]);
           }
         }
       }

       //sqw
       if(!Q.empty())
       {
         std::fill(buffAsComplexVec, buffAsComplexVec+fftSize, std::complex<double>(0.));
         for(unsigned Qidx=0;Qidx<Q.size();Qidx++)
         {
           for(unsigned dim=0;dim<3;dim++)
           {
             for(unsigned k=0;k<m_nFrame;k++)
             {
               double x = Q[Qidx]*trj[k*3+dim];
               double cx(cos(x)), sx(sin(x));
               sqwfftwComplexBuffer[k][0]=cx*window[k];//*modulation[k];
               sqwfftwComplexBuffer[k][1]=sx*window[k];//*modulation[k];
             }

             fftw_execute(fftPlan_c2c);
             fftw_flops(fftPlan_c2c, &tadd, &tmul, &tfma);
             add += tadd;
             mul += tmul;
             fma += tfma;
             for(unsigned j=0;j<frame_used;j++)
             {
               sqwVec[vodsSize*Qidx+j] += sqwfftwComplexBuffer_out[j][0]*sqwfftwComplexBuffer_out[j][0]+sqwfftwComplexBuffer_out[j][1]*sqwfftwComplexBuffer_out[j][1];
             }
           }
         }
       }

       if(moni)
          moni->OneTaskCompleted();
     }

     #pragma omp critical
     {
       fftw_cleanup();
       fftw_destroy_plan(fftPlan_r2c);
       if(!Q.empty())
         fftw_destroy_plan(fftPlan_c2c);
     }

     for(size_t j=0;j<frame_used;++j)
     {
       vdosVec[j]=stableVdos[j].sum();
     }

     // clean up
     fftw_free(realBuffer);
     fftw_free(vdosfftwComplexBuffer);
     fftw_free(sqwfftwComplexBuffer);
     fftw_free(sqwfftwComplexBuffer_out);
     if(moni)
         delete moni;
  }
  //the weight after applying window is unknown at the moment. So use incoherent property for scaling
  unsigned numKpoint = 3;
  double scale = m_nMolecule*numKpoint*m_nFrame; //sqw is calculated for an atom in each molecule, and with 3 dimensions.
  double weight = 1./scale;
  omega = getOmegaVector();


  for(auto &v: sqw.vec)
    v *= weight;


  vdos.dim.resize(1);
  vdos.dim[0]=vodsSize;
  vdos.sanityCheck();

  if(Q.size()!=0)
  {
    sqw.dim.resize(2);
    sqw.dim[0]=Q.size();
    sqw.dim[1]=vodsSize;
    sqw.sanityCheck();
  }

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Number of FFT math operations: additions " << add
        << ", multiplications "<< mul
        << ", multiply-add operations " << fma << std::endl;

  std::cout << "Elapsed time "
        << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
        << "[ms]" << std::endl << std::endl;
}

void Trajectory::structFactSqw(unsigned qSize ,
               Data<double> &structFact, std::vector<double> &scat_length) const
{
  if(scat_length.empty())
    throw std::runtime_error ("scattering length vector empty ");
  if(scat_length.size() != m_nAtomPerMole)
    throw std::runtime_error ("The element number in the scattering length vector should match atom number per mole ");

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  //set the correct size
  //As suggested in G.R. Kneller Comp. Phys. Comm., 1995,
  //input is padded with zero to double the size
  size_t fftSize= atomVdosFFTSize();
  size_t vodsSize =fftSize/2+1;

  structFact.vec.resize(0);
  structFact.vec.resize(qSize, 0.);

  std::vector<double>& structFactVec = structFact.vec;
  fftw_complex *fourierParDen = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*qSize*m_nFrame);
  auto asCXXFourierParDen = reinterpret_cast<std::complex<double>* > (fourierParDen);

  #pragma omp parallel default(none) firstprivate( qSize, fftSize, vodsSize, scat_length) shared( structFactVec, fourierParDen, asCXXFourierParDen)
  {
    if(omp_get_thread_num()==0)
      printf(" structFactSqw uses %d threads\n", omp_get_num_threads());

     double normFact = sqrt(1./(3* m_nAtom*m_nFrame));
     // create pointers
     fftw_complex    *fftwComplexBuffer;
     fftw_plan       fftPlan_c2c;
     std::vector<double> frame;

     fftwComplexBuffer = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*fftSize);
     auto cBufAsCXXComplex = reinterpret_cast<std::complex<double>* > (fftwComplexBuffer);

     // create FFTW plan
     #pragma omp critical
     {
       fftPlan_c2c = fftw_plan_dft_1d(fftSize, fftwComplexBuffer, fftwComplexBuffer, FFTW_FORWARD, FFTW_ESTIMATE);
     }

     Prompt::ProgressMonitor *moni;
     unsigned numThread = omp_get_num_threads();
     unsigned threadID = omp_get_thread_num();
     unsigned threadDoesPrint = 0;
     if(threadID==threadDoesPrint)
         moni = new Prompt::ProgressMonitor("Task structFactSqw", m_nFrame/numThread);

     //structFact
     #pragma omp for simd reduction (+ : structFactVec)
     for(unsigned frameID=0;frameID<m_nFrame;frameID++)
     {
       //load a frame
       readFrame(frameID, frame);
       for(unsigned Qidx=0;Qidx<qSize;Qidx++)
       {
         for(unsigned dim=0;dim<3;dim++)
         {
           StableSum sum_cos_term, sum_sin_term;
           double  unitQ = m_fixed_box_size ? 2*M_PI/(m_box.vec[dim]):2*M_PI/(m_box.vec[frameID*3+dim]);

           for(unsigned n=0;n<m_nAtom;n++)
           {
             double dotprd = (Qidx+1)*unitQ*(frame[n*3+dim]);
             double scattering_length = scat_length[n%m_nAtomPerMole];
             sum_cos_term.add(scattering_length*cos(dotprd));
             sum_sin_term.add(scattering_length*sin(dotprd));
           }
           //update fourier pair density vector, of which the correlation to be calculated in the next loop
           double sum_cos = sum_cos_term.sum()*normFact;
           double sum_sin = sum_sin_term.sum()*normFact;

           fourierParDen[Qidx*m_nFrame+frameID][0] = sum_cos;
           fourierParDen[Qidx*m_nFrame+frameID][1] = sum_sin;
           //update structure factor
           double frameContribution = (sum_cos * sum_cos  + sum_sin * sum_sin );
           structFactVec[Qidx] += frameContribution;
         }
       }
       if(threadID==threadDoesPrint)
          moni->OneTaskCompleted();
     }
  }

  structFact.dim.resize(1);
  structFact.dim[0]=qSize;
  structFact.sanityCheck();

  // sqw.dim.resize(2);
  // sqw.dim[0]=qSize;
  // sqw.dim[1]=vodsSize;
  // sqw.sanityCheck();
  fftw_free(fourierParDen);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "structFactSqw time " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
}



int Trajectory::gcd(int a, int b) const
{
    if (b == 0)
        return a;
    return gcd(b, a % b);
}

unsigned Trajectory::gcd(const std::vector<unsigned> &vec) const
{
  int result=vec[0];
  for(auto v: vec)
  {
    result = gcd(result, v);
  }
  return result;
}

void Trajectory::unwrap(std::vector<double> &atomtrj) const
{
  //fixme: assuming lengths of each dimension are approximately equivelent
  const double L(m_box.vec[0]);
  const double halfL(L*0.5);


  //find atoms outside the box in the first frame
  for(size_t i=0;i<m_nFrame;++i)
  {
    for(size_t dim=0;dim<3;dim++)
    {
      if(atomtrj[i*3 + dim] < 0.)
      {
        atomtrj[i*3 + dim] += m_box.vec[dim];
      }
      else if(atomtrj[i*3 + dim] > L)
      {
        atomtrj[i*3 + dim] -= m_box.vec[dim];
      }
      if(atomtrj[i*3 + dim] < 0. || atomtrj[i*3 + dim] > L)
        throw std::runtime_error ("Corrected postion is still outside the box");
    }
  }

  const double i_L = 1./L;

  for(size_t i=3; i<m_nFrame*3; ++i)
  {
    double diff = atomtrj[i]-atomtrj[i-3];
    if(std::fabs(diff) >  halfL)
    {
      // printf("%d before %g, %g, diff %g, L %g\n", i, atomtrj[i], atomtrj[i-3], diff, L);
      atomtrj[i] -= m_fixed_box_size ? round(diff*i_L)*L : round(diff*i_L)*m_box.vec[i] ;

      if(std::fabs(atomtrj[i]-atomtrj[i-3])>halfL)
      {
        printf("after %g, %g \n \n", atomtrj[i], atomtrj[i-3]);
        throw std::runtime_error ("Correction wrong");
      }
    }
  }
}
