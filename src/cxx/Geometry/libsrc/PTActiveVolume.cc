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

#include "PTActiveVolume.hh"
#include <VecGeom/navigation/BVHNavigator.h>
#include <VecGeom/navigation/NewSimpleNavigator.h>
#include "PTMirrorPhysics.hh"

Prompt::ActiveVolume::ActiveVolume()
:m_geo(vecgeom::GeoManager::Instance()), m_currPV(nullptr),
m_currState(vecgeom::NavigationState::MakeInstance(m_geo.getMaxDepth())),
m_nextState(vecgeom::NavigationState::MakeInstance(m_geo.getMaxDepth()))
{}

Prompt::ActiveVolume::~ActiveVolume()
{
  delete m_currState;
  delete m_nextState;
  std::cout << "Destructed ActiveVolume" << std::endl;
}

Prompt::VolumePhysicsScorer *getLogicalVolumePhysicsScorer(const vecgeom::LogicalVolume &lv)
{
  return (Prompt::VolumePhysicsScorer *)lv.GetUserExtensionPtr();
}

void Prompt::ActiveVolume::locateLogicalVolume(const Vector &p) const
{
  m_currState->Clear();
  auto pv = vecgeom::GlobalLocator::LocateGlobalPoint(m_geo.GetWorld(),
                          *reinterpret_cast<const vecgeom::Vector3D<vecgeom::Precision>*>(&p), *m_currState, true);
  pt_assert_always(pv == m_currState->Top());
}

bool Prompt::ActiveVolume::exitWorld() const
{
  return m_currState->IsOutside();
}

void Prompt::ActiveVolume::setupVolPhysAndGeoTrans()
{
  // Find next step
  // m_currState->Top() gets the placed volume
  m_currPV = m_currState->Top();
  auto &geo = Singleton<GeoManager>::getInstance();
  m_matphysscor = geo.getVolumePhysicsScorer(getVolumeID())->second;

  makeGeoTranslator(); //set up the global to local translator for this volume
}

bool Prompt::ActiveVolume::surfaceReaction(Particle &particle) const
{
  if(hasBoundaryPhyiscs())
  {
    m_matphysscor->boundaryPhysics->sampleFinalState(particle);
    return true;
  }
  else
    return false;
}

void Prompt::ActiveVolume::getNormal(const Vector& pos, Vector &normal) const
{
  const auto &vegpos = *reinterpret_cast<const vecgeom::Vector3D<vecgeom::Precision>*>(&pos);
  auto &vegnormal = *reinterpret_cast<vecgeom::Vector3D<vecgeom::Precision>*>(&normal);
  m_currPV->Normal(vegpos, vegnormal);
}


const Prompt::GeoTranslator& Prompt::ActiveVolume::getGeoTranslator() const
{
  return m_translator;
}

size_t Prompt::ActiveVolume::getVolumeID() const
{
  return m_currPV->GetLogicalVolume()->id();
}

size_t Prompt::ActiveVolume::numSubVolume() const
{
  return m_currPV->GetLogicalVolume()->GetNTotal();
}


std::string Prompt::ActiveVolume::getVolumeName() const
{
  return m_currPV->GetLogicalVolume()->GetName();
}

const vecgeom::VPlacedVolume *Prompt::ActiveVolume::getVolume() const
{
  return m_currPV;
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
          v->score(particle);
    }
  }
}

bool Prompt::ActiveVolume::hasBoundaryPhyiscs() const
{
  return m_matphysscor->boundaryPhysics.use_count();
}

bool Prompt::ActiveVolume::proprogateInAVolume(Particle &particle)
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
