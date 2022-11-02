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

#include "PTNavManager.hh"
#include <VecGeom/navigation/BVHNavigator.h>
#include <VecGeom/navigation/NewSimpleNavigator.h>
#include "PTMirrorPhysics.hh"

Prompt::NavManager::NavManager()
:m_geo(vecgeom::GeoManager::Instance()), m_currPV(nullptr),
m_currState(vecgeom::NavigationState::MakeInstance(m_geo.getMaxDepth())),
m_nextState(vecgeom::NavigationState::MakeInstance(m_geo.getMaxDepth()))
{}

Prompt::NavManager::~NavManager()
{
  delete m_currState;
  delete m_nextState;
  std::cout << "Destructed NavManager" << std::endl;
}

Prompt::VolumePhysicsScorer *getLogicalVolumePhysicsScorer(const vecgeom::LogicalVolume &lv)
{
  return (Prompt::VolumePhysicsScorer *)lv.GetUserExtensionPtr();
}

void Prompt::NavManager::locateLogicalVolume(const Vector &p)
{
  m_currState->Clear();
  auto pv = vecgeom::GlobalLocator::LocateGlobalPoint(m_geo.GetWorld(),
                          {p.x(), p.y(), p.z()}, *m_currState, true);
  assert(pv == m_currState->Top());
}

bool Prompt::NavManager::exitWorld()
{
  return m_currState->IsOutside();
}

void Prompt::NavManager::setupVolumePhysics()
{
  // Find next step
  // m_currState->Top() gets the placed volume
  m_currPV = m_currState->Top();
  auto &geo = Singleton<GeoManager>::getInstance();
  m_matphysscor = geo.getVolumePhysicsScorer(getVolumeID())->second;
}

bool Prompt::NavManager::surfaceReaction(Particle &particle)
{
  if(hasBoundaryPhyiscs())
  {
    auto ptype = m_matphysscor->boundaryPhysics->getType();
    if(ptype==BoundaryPhysics::PhysicsType::Mirror)
    {
      Vector pos(particle.getPosition());
      vecgeom::cxx::Vector3D<double> norm;
      m_currPV->Normal({pos.x(), pos.y(), pos.z()}, norm);
      auto mphys = dynamic_cast<MirrorPhyiscs*>(m_matphysscor->boundaryPhysics.get());
      mphys->setReflectionNormal(*reinterpret_cast<Vector*>(&norm));
    }
    m_matphysscor->boundaryPhysics->sampleFinalState(particle);
    return true;
  }
  else
    return false;
}



size_t Prompt::NavManager::getVolumeID()
{
  return m_currPV->GetLogicalVolume()->id();
}

std::string Prompt::NavManager::getVolumeName()
{
  return m_currPV->GetLogicalVolume()->GetName();
}

const vecgeom::VPlacedVolume *Prompt::NavManager::getVolume()
{
  return m_currPV;
}

void Prompt::NavManager::scoreEntry(Prompt::Particle &particle)
{
  if(m_matphysscor->entry_scorers.size())
  {
    for(auto &v:m_matphysscor->entry_scorers)
    {
      v->score(particle);
    }
  }
}

void Prompt::NavManager::scoreSurface(Prompt::Particle &particle)
{
  auto localposition = particle.getPosition();
  if(m_matphysscor->surface_scorers.size())
  {
    auto loc = m_currState->GlobalToLocal({localposition.x(), localposition.y(), localposition.z()});
    particle.setLocalPosition(Prompt::Vector(loc[0], loc[1], loc[2]));
    for(auto &v:m_matphysscor->surface_scorers)
    {
      v->score(particle);
    }
  }
}

void Prompt::NavManager::scoreAbsorb(Prompt::Particle &particle)
{
  if(m_matphysscor->absorb_scorers.size())
  {
    for(auto &v:m_matphysscor->absorb_scorers)
    {
      v->score(particle);
    }
  }
}

void Prompt::NavManager::scorePropagate(Prompt::Particle &particle)
{
  if(m_matphysscor->propagate_scorers.size())
  {
    for(auto &v:m_matphysscor->propagate_scorers)
    {
      v->score(particle);
    }
  }
}

void Prompt::NavManager::scoreExit(Prompt::Particle &particle)
{
  if(m_matphysscor->exit_scorers.size())
  {
    for(auto &v:m_matphysscor->exit_scorers)
    {
      v->score(particle);
    }
  }
}
bool Prompt::NavManager::hasBoundaryPhyiscs()
{
  return m_matphysscor->boundaryPhysics.use_count();
}

bool Prompt::NavManager::proprogateInAVolume(Particle &particle)
{
  if(!particle.isAlive())
    return false;

  const auto *p = reinterpret_cast<const vecgeom::Vector3D<vecgeom::Precision>*>(&particle.getPosition());
  const auto *dir = reinterpret_cast<const vecgeom::Vector3D<vecgeom::Precision>*>(&particle.getDirection());
  double stepLength = m_matphysscor->bulkPhysics->sampleStepLength(particle);

  //! updates m_nextState to contain information about the next hitting boundary:
  //!   - if a daugher is hit: m_nextState.Top() will be daughter
  //!   - if ray leaves volume: m_nextState.Top() will point to current volume
  //!   - if step limit > step: m_nextState == in_state
  //!   ComputeStep is essentialy equal to ComputeStepAndPropagatedState without the relaction part
  double step = m_currPV->GetLogicalVolume()->GetNavigator()->ComputeStepAndPropagatedState(*p, *dir, stepLength, *m_currState, *m_nextState);
  std::swap(m_currState, m_nextState);

  bool sameVolume (step == stepLength);
  // assert(stepLength >= step);
  if(stepLength < step)
    PROMPT_THROW2(BadInput, "stepLength < step " << stepLength << " " << step << "\n");

  //Move next step
  const double resolution = 10*vecgeom::kTolerance; //this value should be in sync with the geometry tolerance
  particle.moveForward(sameVolume ? step : (step + resolution) );

  if(sameVolume)
  {
    //sample the interaction at the location
    m_matphysscor->bulkPhysics->sampleFinalState(particle, step, false);
    return true;
  }
  else
  {
    m_matphysscor->bulkPhysics->sampleFinalState(particle, step, true);
    return false;
  }
}
