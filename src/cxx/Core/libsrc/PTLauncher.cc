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

#include "PTLauncher.hh"

#include "PTGeoManager.hh"
#include "PTNavManager.hh"
#include "PTMath.hh"
#include "PTParticle.hh"
#include "PTProgressMonitor.hh"
#include "PTSimpleThermalGun.hh"
#include "PTNeutron.hh"

#include "NCrystal/NCrystal.hh"
class SingletonPTRandWrapper : public NCrystal::RNGStream{
public:
  SingletonPTRandWrapper()
  :NCrystal::RNGStream(), m_ptrng(Prompt::Singleton<Prompt::SingletonPTRand>::getInstance())
  {}
  virtual ~SingletonPTRandWrapper() override {}

  double actualGenerate() override {return m_ptrng.generate(); }

  //For the sake of example, we wrongly claim that this generator is safe and
  //sensible to use multithreaded (see NCRNG.hh for how to correctly deal with
  //MT safety, RNG states, etc.):
  bool useInAllThreads() const override { return true; }
private:
  Prompt::SingletonPTRand &m_ptrng;
};


Prompt::Launcher::Launcher()
{

}


Prompt::Launcher::~Launcher()
{
  printLogo();
}

void Prompt::Launcher::loadGeometry(const std::string &geofile)
{
  //This checks that the included NCrystal headers and the linked NCrystal
  //library are from the same release of NCrystal:
  NCrystal::libClashDetect();

  //set the generator for ncrystal
  NCrystal::setDefaultRNG(NCrystal::makeSO<SingletonPTRandWrapper>());

  //load geometry
  auto &geoman = Singleton<GeoManager>::getInstance();
  geoman.loadFile(geofile.c_str());
  if(geoman.m_gun.use_count())
    m_gun = geoman.m_gun;
}


void Prompt::Launcher::go(uint64_t numParticle, double printPrecent, bool recordTrj)
{
  //set the seed for the random generator
  auto &rng = Singleton<SingletonPTRand>::getInstance();

  //create navigation manager
  auto &navman = Singleton<NavManager>::getInstance();

  if(!m_gun.use_count())
  {
    std::cout << "PrimaryGun is not set, fallback to the neutron SimpleThermalGun\n";
    m_gun = std::make_shared<SimpleThermalGun>(Neutron());
  }

  DeltaParticle dltpar;

  ProgressMonitor moni("Prompt simulation", numParticle, printPrecent);
  for(size_t i=0;i<numParticle;i++)
  {
    //double ekin, const Vector& dir, const Vector& pos
    auto particle = m_gun->generate();
    dltpar.setLastParticle(particle);
    // std::cout << "ekin " << particle.getEKin() << std::endl;

    if(recordTrj)
    {
      std::vector<Vector> tmp;
      tmp.reserve(m_trajectory.size());
      m_trajectory.swap(tmp);
    }

    //! allocate the point in a volume
    navman.locateLogicalVolume(particle.getPosition());
    while(!navman.exitWorld() && particle.isAlive())
    {
      if(recordTrj)
      {
        m_trajectory.push_back(particle.getPosition());
      }

      //! first step of a particle in a volume
      // std::cout << navman.getVolumeName() << " " << particle.getPosition() << std::endl;
      navman.setupVolumePhysics();
      navman.scoreSurface(particle.getPosition(), particle.getWeight());

      //if reflected
      if(navman.surfacePhysics(particle))
      {
        // std::cout << "reflection weight " << particle.getWeight() << "\n";
      }
      navman.scoreEntry(particle);

      //! the next while loop, particle should move in the same volume
      while(navman.proprogateInAVolume(particle, 0) )
      {
        // score if any scoror is available
        if(navman.hasPropagateScoror())
        {
          dltpar.calcDeltaParticle(particle);
          navman.scorePropagate(particle, dltpar);
        }
        if(recordTrj)
          m_trajectory.push_back(particle.getPosition());
      }
      navman.scoreExit(particle);
    }

    if(!navman.exitWorld() && !particle.isAlive())
    {
      if(particle.getKillType()==Particle::ABSORB)
      {
        // std::cout << "Logging ABSORB  " << std::endl;
        // navman.scoreAbsorb(particle);
      }
    }

    if(recordTrj)
    {
      m_trajectory.push_back(particle.getPosition());
    }
    moni.OneTaskCompleted();
  }
}
