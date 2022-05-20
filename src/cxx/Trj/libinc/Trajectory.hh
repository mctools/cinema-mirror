#ifndef Trajectory_hh
#define Trajectory_hh

#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
#include <memory>
#include "Fourier.hh"

#include "Data.hh"

class Trajectory
{
public:
  Trajectory(std::string fileName);
  virtual ~Trajectory();
  virtual void readAtomTrj(unsigned atomOffset, std::vector<double> &trj) const = 0;
  virtual const double* const readAtomTrj(unsigned atomOffset) const = 0;
  virtual void readFrame(unsigned frameID, std::vector<double> &frame) const = 0;
  size_t atomVdosFFTSize() const;

  //calculates the angular frequency that is compatible with the results of vDosSqw and structFactSqw
  Data<double> unweightedFrequency() const;
  Data<double> getOmegaVector() const;

  void vDosSqw(unsigned firstAtomPos, Data<double> &vdos,  Data<double> &omega,
                 const std::vector<double> &Q, Data<double> &sqw) const;

  void coherentDynamicsFactor(unsigned qSize, Data<double> &sqw, std::vector<double> &scat_length) const;
  void structFactSqw(unsigned qSize, Data<double> &structFact, std::vector<double> &scat_length) const;

  //serial for validation purpose
  void atomVdos(unsigned atomOffset, std::vector<double> &vdos, bool accumulate=false) const;
  double getDeltaT() const {return m_deltaT;}
  double getQResolution() const {return m_qResolution;};
  double getNumFrame() const {return m_nFrame;}
  double getNumAtom() const {return m_nAtom;}
  double getNumMolecule() const {return m_nMolecule;}
  double getNumAtomPerMolecule() const {return m_nAtomPerMole;}


protected:
  std::string m_fileName;
  bool m_fixed_box_size;
  long unsigned m_nFrame, m_nAtom, m_nMolecule, m_nAtomPerMole;
  double m_deltaT; //in the unit of second
  double m_qResolution; //in the unit of Aa
  std::vector<unsigned> m_atomType, m_atomCount;
  Data<unsigned> m_species;
  Data<double> m_time, m_box;

  void unwrap(std::vector<double> &atomtrj) const;
  unsigned gcd(const std::vector<unsigned> &vec) const;
  int gcd(int a, int b) const;


};

#endif
