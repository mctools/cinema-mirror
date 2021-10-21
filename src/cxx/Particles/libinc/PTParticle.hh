#ifndef Prompt_Particale_hh
#define Prompt_Particale_hh

#include "PTVector.hh"
#include "PTMath.hh"
#include "PTUnitSystem.hh"

//! Particle is neutron with pgd code of 2112 by defult. Proton (2212) is also supported.
//! fixme: support Gamma (22) as well.
namespace Prompt {
  class Particle {
  public:
    Particle();
    Particle(double ekin, const Vector& dir, const Vector& pos);
    virtual ~Particle(){};

    virtual void moveForward(double length);
    virtual void changeDirectionTo(const Vector& dir);

    void changeEKinTo(double ekin);
    void changePositionTo(const Vector& pos);
    Vector &getDirection() {return m_dir; }
    Vector &getPosition() {return m_pos; }
    double getTime() {return m_time;}
    double getEKin() {return m_ekin;}

    virtual double calcSpeed() const;
  protected:
    double m_ekin, m_time;
    Vector m_dir, m_pos;
    unsigned m_pgd;
    double m_rest_mass;
  };
}


inline Prompt::Particle::Particle()
  :m_ekin(0.), m_time(0.), m_dir(), m_pos(), m_pgd(0), m_rest_mass(0.)
{}

inline Prompt::Particle::Particle(double ekin, const Vector& dir, const Vector& pos)
  :m_ekin(ekin), m_time(0.), m_dir(dir), m_pos(pos), m_pgd(0) ,m_rest_mass(0)
{}

inline void Prompt::Particle::moveForward(double length)
{
  m_pos += m_dir*length;
  if(m_ekin)
    m_time += length/calcSpeed();
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

inline double Prompt::Particle::calcSpeed() const
{
    return std::sqrt(2*m_ekin/m_rest_mass);
}

#endif
