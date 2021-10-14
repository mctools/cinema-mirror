#ifndef Prompt_Particale_hh
#define Prompt_Particale_hh

#include "PTVector.hh"
#include "PTMath.hh"

//! Particle is neutron with pgd code of 2112 by defult. Proton (2212) is also supported.
//! m_erest is in the unit of eV*c^2
//! fixme: support Gamma (22) as well.
namespace Prompt {
  class Particle {
  public:
    Particle();
    Particle(double ekin, const Vector& dir, const Vector& pos);
    virtual ~Particle(){};

    void moveForward(double length);
    double moveForwardTimed(double length);
    void changeEKinTo(double ekin);
    void changeDirectionTo(const Vector& dir);
    void changePositionTo(const Vector& pos);

    virtual double calcSpeed() const;
  protected:
    double m_ekin; /// mass is in the unit of MeV/const_c^2
    Vector m_dir, m_pos;
    unsigned m_pgd;
    double m_rest_mass; /// Relativistic mass = (E_tot + E_rest) / E_rest * rest_mass
  };
}


inline Prompt::Particle::Particle()
  :m_ekin(0.), m_dir(), m_pos(), m_pgd(0), m_rest_mass(0.)
{}

inline Prompt::Particle::Particle(double ekin, const Vector& dir, const Vector& pos)
  :m_ekin(ekin), m_dir(dir), m_pos(pos), m_pgd(0) ,m_rest_mass(0)
{}

inline void Prompt::Particle::moveForward(double length)
{
  m_pos += m_dir*length;
}

inline double Prompt::Particle::moveForwardTimed(double length)
{
  moveForward(length);
  double speed = calcSpeed();
  return length/speed;
}

inline void Prompt::Particle::changeEKinTo(double ekin)
{
  m_ekin = ekin;
}

inline void Prompt::Particle::changePositionTo(const Vector& pos)
{
  m_pos = pos;
}

inline void Prompt::Particle::changeDirectionTo(const Vector& dir)
{
  m_dir = dir;
}

#include <iostream>

//! for time-dependent system simulation
inline double Prompt::Particle::calcSpeed() const
{
  //fixme: tune a better threshold
  if(m_ekin>0.11*const_neutron_rest_mass)
  {
    double gamma = 1 + m_ekin/m_rest_mass;
    double temp = 1/gamma;
    double beta = std::sqrt(1-temp*temp);
    //printf("Relativistic %.16g \n", const_c*beta);
    return const_c*beta;
  }
  else
  {
    // printf("normal %.16g\n", std::sqrt(2*m_ekin/m_rest_mass*const_c*const_c));
    return std::sqrt(2*m_ekin/m_rest_mass*const_cc);
  }
}

#endif
