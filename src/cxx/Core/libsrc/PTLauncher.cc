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

#include "PTGeoLoader.hh"
#include "PTActiveVolume.hh"
#include "PTStackManager.hh"
#include "PTMath.hh"
#include "PTParticle.hh"
#include "PTProgressMonitor.hh"
#include "PTIsotropicGun.hh"
#include "PTNeutron.hh"

#include "PTPython.hh"

Prompt::Launcher::Launcher()
{
  pt_enable_prompt();
}


Prompt::Launcher::~Launcher()
{
  printLogo();
}

void Prompt::Launcher::loadGeometry(const std::string &geofile)
{
  //load geometry
  auto &geoman = Singleton<GeoLoader>::getInstance();
  geoman.initFromGDML(geofile.c_str());
  if(geoman.m_gun.use_count())
    m_gun = geoman.m_gun;
}



void Prompt::Launcher::steupFakeGeoPhyisc() //for c++ debug
{

  //load geometry
  auto &geoman = Singleton<GeoLoader>::getInstance();
  // geoman.steupFakePhyisc();
}


void Prompt::Launcher::go(uint64_t numParticle, double printPrecent, bool recordTrj, bool timer)
{
  //set the seed for the random generator
  auto &rng = Singleton<SingletonPTRand>::getInstance();

  //create navigation manager
  auto &activeVolume = Singleton<ActiveVolume>::getInstance();

  auto &stackManager = Singleton<StackManager>::getInstance();

  if(!m_gun.use_count())
  {
    std::cout << "PrimaryGun is not set, fallback to the neutron IsotropicGun\n";
    m_gun = std::make_shared<IsotropicGun>(Neutron(), 0.0253, Vector{0,0,0}, Vector{1,0,0});
  }

  ProgressMonitor *moni=nullptr;
  if(timer)
    moni = new ProgressMonitor("Prompt simulation", numParticle, printPrecent);
  for(size_t i=0;i<numParticle;i++)
  {
    //add a primary particle into the stack
    stackManager.add(m_gun->generate());

    while(!stackManager.empty())
    {
      auto particle = *(stackManager.pop()).get();

      if(recordTrj)
      {
        std::vector<Vector> tmp;
        tmp.reserve(m_trajectory.size());
        m_trajectory.swap(tmp);
      }

      //! allocate the point in a volume
      activeVolume.locateActiveVolume(particle.getPosition());
      while(!activeVolume.exitWorld() && particle.isAlive())
      {
        if(recordTrj)
        {
          m_trajectory.push_back(particle.getPosition());
        }

        //! first step of a particle in a volume
        // std::cout << activeVolume.getVolumeName() << " " << particle.getPosition() << std::endl;
        activeVolume.setupVolPhysAndGeoTrans();
        activeVolume.scoreSurface(particle);

        //if reflected, absorbed, transmitted
        if(activeVolume.surfaceReaction(particle))
        {
          // std::cout << "reflection weight " << particle.getWeight() << "\n";
        }
        activeVolume.scoreEntry(particle);

        //! within the next while loop, particle should move in the same volume
        while(activeVolume.proprogateInAVolume(particle) )
        {
          // score if any scorer is available
          if(activeVolume.hasPropagateScorer())
          {
            activeVolume.scorePropagate(particle);
          }
          if(recordTrj)
            m_trajectory.push_back(particle.getPosition());
        }
        activeVolume.scoreExit(particle);
      }

      if(!particle.isAlive())
      {
        if(particle.getKillType()==Particle::KillType::ABSORB)
        {
          activeVolume.scoreAbsorb(particle);
        }
      }

      if(recordTrj)
      {
        m_trajectory.push_back(particle.getPosition());
      }
    }
    if(timer)
      moni->OneTaskCompleted();
  }

  if(timer)
  {   
    delete moni;
    Singleton<ResourceManager>::getInstance().writeScorer2Disk();
  }
}
