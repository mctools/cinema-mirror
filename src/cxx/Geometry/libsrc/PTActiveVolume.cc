////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//  This file is part of Prompt (see https://gitlab.com/xxcai1/Prompt)        //
//                                                                            //
//  Copyright 2021-2024 Prompt developers                                     //
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

#include "PTActiveVolume.hh"
#include <VecGeom/navigation/BVHNavigator.h>
#include <VecGeom/navigation/NewSimpleNavigator.h>
#include "PTMirror.hh"

Prompt::ActiveVolume::ActiveVolume()
:m_geo(vecgeom::GeoManager::Instance()), 
m_currState(vecgeom::NavigationState::MakeInstance(0)),
m_nextState(vecgeom::NavigationState::MakeInstance(0))
{}

Prompt::ActiveVolume::~ActiveVolume()
{
  delete m_currState;
  delete m_nextState;
  std::cout << "Destructed ActiveVolume" << std::endl;
}


void Prompt::ActiveVolume::setup()
{
  if(m_currState)
    delete m_currState;
  if(m_nextState)
    delete m_nextState;

  m_currState = vecgeom::NavigationState::MakeInstance(m_geo.getMaxDepth());
  m_nextState = vecgeom::NavigationState::MakeInstance(m_geo.getMaxDepth());
}

// Prompt::VolumePhysicsScorer *getLogicalVolumePhysicsScorer(const vecgeom::LogicalVolume &lv)
// {
//   return (Prompt::VolumePhysicsScorer *)lv.GetUserExtensionPtr();
// }

bool Prompt::ActiveVolume::locateActiveVolume(const Vector &p) const
{
  m_currState->Clear();
  m_nextState->Clear();

  auto pv = vecgeom::GlobalLocator::LocateGlobalPoint(m_geo.GetWorld(),
                          *reinterpret_cast<const vecgeom::Vector3D<vecgeom::Precision>*>(&p), *m_currState, true);

  if(!pv)
    std::cout << "particle at " << p << " is outside the world" << std::endl;
  return exitWorld() || !pv;
} 

bool Prompt::ActiveVolume::exitWorld() const
{
  return m_currState->IsOutside();
}

void Prompt::ActiveVolume::setupVolPhysAndGeoTrans()
{
  // Find next step
  // m_currState->Top() gets the placed volume
  m_matphysscor = Singleton<ResourceManager>::getInstance().getVolumePhysicsScorer(getVolumeID());
  if(!m_matphysscor)
    PROMPT_THROW2(CalcError, "Volume with id " << getVolumeID() << " is not found in the resource manager");

  // auto &geo = Singleton<GeoManager>::getInstance();
  // m_matphysscor = geo.getVolumePhysicsScorer(getVolumeID())->second;

  makeGeoTranslator(); //set up the global to local translator for this volume
}

bool Prompt::ActiveVolume::surfaceReaction(Particle &particle) const
{
  if(hasBoundaryPhyiscs())
  {
    m_matphysscor->surfaceProcess->sampleFinalState(particle);
    return true;
  }
  else
    return false;
}

void Prompt::ActiveVolume::getNormal(const Vector& pos, Vector &normal) const
{
  const auto &vegpos = *reinterpret_cast<const vecgeom::Vector3D<vecgeom::Precision>*>(&pos);
  auto &vegnormal = *reinterpret_cast<vecgeom::Vector3D<vecgeom::Precision>*>(&normal);
   m_currState->Top()->Normal(vegpos, vegnormal);
}


const Prompt::GeoTranslator& Prompt::ActiveVolume::getGeoTranslator() const
{
  return m_translator;
}

size_t Prompt::ActiveVolume::getVolumeID() const
{
  return  m_currState->Top()->GetLogicalVolume()->id();
}

size_t Prompt::ActiveVolume::numSubVolume() const
{
  return  m_currState->Top()->GetLogicalVolume()->GetNTotal();
}

double Prompt::ActiveVolume::distanceToOut(const Vector& loc_pos, const Vector &loc_dir) const
{
  return  m_currState->Top()->DistanceToOut(*reinterpret_cast<const vecgeom::Vector3D<vecgeom::Precision>*>(&loc_pos),
                                 *reinterpret_cast<const vecgeom::Vector3D<vecgeom::Precision>*>(&loc_dir));
}


std::string Prompt::ActiveVolume::getVolumeName() const
{
  return  m_currState->Top()->GetLogicalVolume()->GetName();
}

const vecgeom::VPlacedVolume *Prompt::ActiveVolume::getVolume() const
{
  return  m_currState->Top();
}


void Prompt::ActiveVolume::makeGeoTranslator()
{
  auto& trans = m_translator.getTransformMatrix();
  m_currState->DeltaTransformation(1, trans); //this is the difference between the volume and the world
}


void Prompt::ActiveVolume::scoreEntry(Prompt::Particle &particle)
{
  if(m_matphysscor->entry_scorers.size())
  {
    for(auto &v:m_matphysscor->entry_scorers)
    {
      v->score(particle);
    }
  }
}

void Prompt::ActiveVolume::scoreSurface(Prompt::Particle &particle)
{
  auto localposition = particle.getPosition();   
  if(m_matphysscor->surface_scorers.size())
  {
    for(auto &v:m_matphysscor->surface_scorers)
    {
      v->score(particle);
    }
  }
}

void Prompt::ActiveVolume::scoreAbsorb(Prompt::Particle &particle)
{
  if(m_matphysscor->absorb_scorers.size())
  {
    for(auto &v:m_matphysscor->absorb_scorers)
    {
      v->score(particle);
    }
  }
}

void Prompt::ActiveVolume::scorePropagate(Prompt::Particle &particle)
{
  if(m_matphysscor->propagate_scorers.size())
  {
    for(auto &v:m_matphysscor->propagate_scorers)
    {
      v->score(particle);
    }
  }
}

void Prompt::ActiveVolume::scoreExit(Prompt::Particle &particle)
{
  if(m_matphysscor->exit_scorers.size())
  {
    for(auto &v:m_matphysscor->exit_scorers)
    {
      // to act along with the Prompt::ScorerRotatingObj::score method,
      // disable the effective energy and direction, by setting the direction to null
      if(particle.hasEffEnergy())
          particle.setEffDirection(Vector());
      else
      {
        v->score(particle);
      }
    }
  }
}

bool Prompt::ActiveVolume::hasBoundaryPhyiscs() const
{
  return m_matphysscor->surfaceProcess.use_count();
}

bool Prompt::ActiveVolume::proprogateInAVolume(Particle &particle)
{
  if(!particle.isAlive())
    return false;

  const auto *p = reinterpret_cast<const vecgeom::Vector3D<vecgeom::Precision>*>(&particle.getPosition());
  const auto *dir = reinterpret_cast<const vecgeom::Vector3D<vecgeom::Precision>*>(&particle.getDirection());
  double stepLength = m_matphysscor->bulkMaterialProcess->sampleStepLength(particle);

  //! updates m_nextState to contain information about the next hitting boundary:
  //!   - if a daugher is hit: m_nextState.Top() will be daughter
  //!   - if ray leaves volume: m_nextState.Top() will point to current volume
  //!   - if step limit > step: m_nextState == in_state
  //!   ComputeStep is essentialy equal to ComputeStepAndPropagatedState without the relaction part
  
  double safety (0.);

  double step =  m_currState->Top()->GetLogicalVolume()->GetNavigator()->ComputeStepAndSafetyAndPropagatedState(*p, *dir, stepLength, *m_currState, *m_nextState, true, safety);

  if(!step && safety==-1.)
  {
    std::cout << "in proprogateInAVolume bulkMaterialProcess->getName() " 
    << m_matphysscor->bulkMaterialProcess->getName() << " steplength " << stepLength << ", step to  boundary "  << step 
    << ", safety is "  << safety << "\n" ;
    PROMPT_THROW2(CalcError, "Vecgeom is unable to computer the distance to the next boundary for logical volume id " << getVolumeID());
  }


  bool sameVolume (m_currState->Top() == m_nextState->Top());
    
  if(stepLength < step)
    PROMPT_THROW2(CalcError, "stepLength < step " << stepLength << " " << step << "\n");

  //Move next step
  const double resolution = 10*vecgeom::kTolerance; //this value should be in sync with the geometry tolerance
  particle.moveForward(sameVolume ? step : (step + resolution) );

  m_matphysscor->bulkMaterialProcess->sampleFinalState(particle, step, !sameVolume);
  if(!sameVolume)
  {
    #ifdef DEBUG_PTS
      std::cout << "Exiting volume " << getVolumeID() << std::endl;
    #endif
    scoreExit(particle);  //score exit before activeVolume changes, otherwise physical volume id and scorer id may be inconsistent.
    std::swap(m_currState, m_nextState);
  }
  //sample the interaction at the location
  return sameVolume;

}
