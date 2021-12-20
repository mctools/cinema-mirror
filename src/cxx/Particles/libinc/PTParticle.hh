#ifndef Prompt_Particale_hh
#define Prompt_Particale_hh

////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//  This file is part of Prompt (see https://gitlab.com/xxcai1/Prompt)        //
//                                                                            //
//  Copyright 2021-2022 Prompt developers                                     //
//                                                                            //
//  Licensed under the Apache License, Version 2.0 (the "License");           //
//  you may not use this file except in compliance with the License.          //
//  You may obtain a copy of the License at                                   //
//                                                                            //
//      http://www.apache.org/licenses/LICENSE-2.0                            //
//                                                                            //
//  Unless required by applicable law or agreed to in writing, software       //
//  distributed under the License is distributed on an "AS IS" BASIS,         //
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  //
//  See the License for the specific language governing permissions and       //
//  limitations under the License.                                            //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include "PTVector.hh"
#include "PTMath.hh"
#include "PTUnitSystem.hh"

//! Particle is neutron with pgd code of 2112 by defult. Proton (2212) is also supported.
//! fixme: support Gamma (22) as well.
namespace Prompt {

  class Particle {
    friend class PrimaryGun;
  public:
    Particle();
    Particle(double ekin, const Vector& dir, const Vector& pos);
    virtual ~Particle(){};

    virtual void moveForward(double length);
    virtual void setDirection(const Vector& dir);

    void setEKin(double ekin);
    void setPosition(const Vector& pos);
    Vector &getDirection() { return m_dir; }
    Vector &getPosition() { return m_pos; }
    double getTime() { return m_time; }
    double getEKin() { return m_ekin; }
    double getWeight() { return m_weight; }
    void scaleWeight(double factor) { m_weight *= factor; }
    unsigned getEventID() { return m_eventid; }

    void kill();
    bool isAlive();

    virtual double calcSpeed() const;
  protected:
    friend class DeltaParticle;
    double m_ekin, m_time;
    Vector m_dir, m_pos;
    unsigned m_pgd;
    double m_weight;
    double m_rest_mass;
    bool m_alive;
    unsigned m_eventid, m_id, m_parentid;
  };

  struct DeltaParticle {
    double dlt_ekin, dlt_time;
    Vector dlt_dir, dlt_pos;
    Particle lastParticle;
    void setLastParticle(const Particle &p)
    {
      lastParticle=p;
    }

    void calcDeltaParticle(const Particle &p)
    {
      dlt_ekin=p.m_ekin-lastParticle.m_ekin;
      dlt_time=p.m_time-lastParticle.m_time;
      dlt_dir=p.m_dir-lastParticle.m_dir;
      dlt_pos=p.m_pos-lastParticle.m_pos;
      lastParticle = p;
    }
  };

}


inline Prompt::Particle::Particle()
  :m_ekin(0.), m_time(0.), m_dir(), m_pos(), m_pgd(0),
  m_weight(1.), m_rest_mass(0.), m_alive(true), m_eventid(0), m_id(0), m_parentid(0)
{}

inline Prompt::Particle::Particle(double ekin, const Vector& dir, const Vector& pos)
  :m_ekin(ekin), m_time(0.), m_dir(dir), m_pos(pos), m_pgd(0),
  m_weight(1.), m_rest_mass(0), m_alive(true), m_eventid(0), m_id(0), m_parentid(0)
{
  m_dir.normalise();
}

inline void Prompt::Particle::moveForward(double length)
{
  m_pos += m_dir*length;
  if(m_ekin)
    m_time += length/calcSpeed();
}

inline void Prompt::Particle::setEKin(double ekin)
{
  m_ekin = ekin;
}

inline void Prompt::Particle::setPosition(const Vector& pos)
{
  m_pos = pos;
}

inline void Prompt::Particle::setDirection(const Vector& dir)
{
  m_dir = dir;
}

inline double Prompt::Particle::calcSpeed() const
{
    return std::sqrt(2*m_ekin/m_rest_mass);
}

#endif
