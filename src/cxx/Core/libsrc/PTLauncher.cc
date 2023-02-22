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
#include "PTActiveVolume.hh"
#include "PTStackManager.hh"
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


//////////////////////////////////////////////////////////////////////////////////////////////////

#include "VecGeom/base/Config.h"
#include "VecGeom/benchmarking/NavigationBenchmarker.h"
#include "VecGeom/volumes/utilities/VolumeUtilities.h"

#include "VecGeom/management/GeoManager.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/volumes/Orb.h"
#include "VecGeom/volumes/Trapezoid.h"

using namespace VECGEOM_NAMESPACE;

VPlacedVolume *fakeGeometry()
{

  UnplacedBox *worldUnplaced      = new UnplacedBox(10, 10, 10);
  UnplacedTrapezoid *trapUnplaced = new UnplacedTrapezoid(4, 0, 0, 4, 4, 4, 0, 4, 4, 4, 0);
  UnplacedBox *boxUnplaced        = new UnplacedBox(2, 2, 2);
  UnplacedOrb *orbUnplaced        = new UnplacedOrb(2.8);

  LogicalVolume *world = new LogicalVolume("world", worldUnplaced);
  LogicalVolume *trap  = new LogicalVolume("trap", trapUnplaced);
  LogicalVolume *box   = new LogicalVolume("box", boxUnplaced);
  LogicalVolume *orb   = new LogicalVolume("orb", orbUnplaced);

  Transformation3D *ident = new Transformation3D(0, 0, 0, 0, 0, 0);
  orb->PlaceDaughter("orb1", box, ident);
  trap->PlaceDaughter("box1", orb, ident);

  Transformation3D *placement1 = new Transformation3D(5, 5, 5, 0, 0, 0);
  Transformation3D *placement2 = new Transformation3D(-5, 5, 5, 0, 0, 0);   // 45,  0,  0);
  Transformation3D *placement3 = new Transformation3D(5, -5, 5, 0, 0, 0);   // 0, 45,  0);
  Transformation3D *placement4 = new Transformation3D(5, 5, -5, 0, 0, 0);   // 0,  0, 45);
  Transformation3D *placement5 = new Transformation3D(-5, -5, 5, 0, 0, 0);  // 45, 45,  0);
  Transformation3D *placement6 = new Transformation3D(-5, 5, -5, 0, 0, 0);  // 45,  0, 45);
  Transformation3D *placement7 = new Transformation3D(5, -5, -5, 0, 0, 0);  // 0, 45, 45);
  Transformation3D *placement8 = new Transformation3D(-5, -5, -5, 0, 0, 0); // 45, 45, 45);

  world->PlaceDaughter("trap1", trap, placement1);
  world->PlaceDaughter("trap2", trap, placement2);
  world->PlaceDaughter("trap3", trap, placement3);
  world->PlaceDaughter("trap4", trap, placement4);
  world->PlaceDaughter("trap5", trap, placement5);
  world->PlaceDaughter("trap6", trap, placement6);
  world->PlaceDaughter("trap7", trap, placement7);
  world->PlaceDaughter("trap8", trap, placement8);

  VPlacedVolume *w = world->Place();
  GeoManager::Instance().SetWorld(w);
  GeoManager::Instance().CloseGeometry();

  // cleanup
  delete ident;
  delete placement1;
  delete placement2;
  delete placement3;
  delete placement4;
  delete placement5;
  delete placement6;
  delete placement7;
  delete placement8;
  return w;
}

//////////////////////////////////////////////////////////////////////////////////////////////////


void Prompt::Launcher::steupFakeGeoPhyisc() //for c++ debug
{
  fakeGeometry();
  //This checks that the included NCrystal headers and the linked NCrystal
  //library are from the same release of NCrystal:
  NCrystal::libClashDetect();

  //set the generator for ncrystal
  NCrystal::setDefaultRNG(NCrystal::makeSO<SingletonPTRandWrapper>());

  //load geometry
  auto &geoman = Singleton<GeoManager>::getInstance();
  geoman.steupFakePhyisc();
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
    std::cout << "PrimaryGun is not set, fallback to the neutron SimpleThermalGun\n";
    m_gun = std::make_shared<SimpleThermalGun>(Neutron());
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
    auto &geoman = Singleton<GeoManager>::getInstance();
    geoman.writeScorer2Disk();
  }
}
