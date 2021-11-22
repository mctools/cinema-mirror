
#include "PTNavManager.hh"
#include <VecGeom/navigation/BVHNavigator.h>
#include <VecGeom/navigation/NewSimpleNavigator.h>

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

Prompt::VolumePhysicsScoror *getLogicalVolumePhysicsScoror(const vecgeom::LogicalVolume &lv)
{
  return (Prompt::VolumePhysicsScoror *)lv.GetUserExtensionPtr();
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
  m_matphysscor = geo.getVolumePhysicsScoror(getVolumeID())->second;
}

bool Prompt::NavManager::surfacePhysics(Particle &particle)
{
  if(hasMirrorPhyiscs())
  {
    Vector pos(particle.getPosition());
    vecgeom::cxx::Vector3D<double> norm;
    m_currPV->Normal({pos.x(), pos.y(), pos.z()}, norm);
    double eout(0), scaleWeigh(0);
    Vector ptNorm{norm[0], norm[1], norm[2]};
    m_matphysscor->mirrorPhysics->generate(particle.getEKin(),
    particle.getDirection(), eout, ptNorm, scaleWeigh);
    particle.setDirection(ptNorm);
    particle.scaleWeight(scaleWeigh);
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
  if(m_matphysscor->entry_scorors.size())
  {
    for(auto &v:m_matphysscor->entry_scorors)
    {
      v->score(particle);
    }
  }
}

void Prompt::NavManager::scorePropagate(Prompt::Particle &particle, const DeltaParticle &dltpar)
{
  if(m_matphysscor->propagate_scorors.size())
  {
    for(auto &v:m_matphysscor->propagate_scorors)
    {
      v->score(particle, dltpar);
    }
  }
}

void Prompt::NavManager::scoreExit(Prompt::Particle &particle)
{
  if(m_matphysscor->exit_scorors.size())
  {
    for(auto &v:m_matphysscor->exit_scorors)
    {
      v->score(particle);
    }
  }
}
bool Prompt::NavManager::hasMirrorPhyiscs()
{
  return m_matphysscor->mirrorPhysics.use_count();
}

bool Prompt::NavManager::proprogateInAVolume(Particle &particle, bool verbose )
{
  if(!particle.isAlive())
    return false;

  Vector &p = particle.getPosition();
  Vector &dir = particle.getDirection();
  if (verbose) {
    std::cout << m_currPV->GetLogicalVolume()->GetName() << ", id " << m_currPV->GetLogicalVolume()->id() << std::endl;
    std::cout << "initial conditions: pos " << p << " , dir "  << dir  << " ekin " << particle.getEKin() << std::endl;
  }

  double stepLength = m_matphysscor->physics->sampleStepLength(particle.getEKin(), dir);

  //! updates m_nextState to contain information about the next hitting boundary:
  //!   - if a daugher is hit: m_nextState.Top() will be daughter
  //!   - if ray leaves volume: m_nextState.Top() will point to current volume
  //!   - if step limit > step: m_nextState == in_state
  //!   ComputeStep is essentialy equal to ComputeStepAndPropagatedState without the relaction part
  double step = m_currPV->GetLogicalVolume()->GetNavigator()->ComputeStepAndPropagatedState({p.x(), p.y(), p.z()}, {dir.x(), dir.y(), dir.z()}, stepLength, *m_currState, *m_nextState);
  std::swap(m_currState, m_nextState);

  bool sameVolume (step == stepLength);
  assert(stepLength >= step);
  if (verbose && !sameVolume) { std::cout << "hitDaugherBoundary\n";}

  //Move next step
  const double resolution = 0; //fixme: this value should be in sync with the geometry tolerance
  particle.moveForward(step + (sameVolume ? 0 : resolution) );

  if (verbose) {
    std::cout << "scattered conditions: pos " << p << " , dir "  << dir  << " ekin " << particle.getEKin() << " step " << step << std::endl << std::endl;
  }

  if(sameVolume)
  {
    //sample the interaction at the location
    double final_ekin(0);
    Vector final_dir;
    double scaleWeight;
    m_matphysscor->physics->sampleFinalState(particle.getEKin(), dir, final_ekin, final_dir, scaleWeight);
    if(final_ekin==-1.)
      particle.kill();
    else
    {
      particle.setEKin(final_ekin);
      particle.setDirection(final_dir);
      particle.scaleWeight(scaleWeight);
    }
    return true;
  }
  else
    return false;
}
