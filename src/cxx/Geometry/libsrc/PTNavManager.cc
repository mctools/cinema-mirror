
#include "PTNavManager.hh"
#include <VecGeom/navigation/BVHNavigator.h>
#include <VecGeom/navigation/NewSimpleNavigator.h>

Prompt::NavManager::NavManager()
:m_geo(vecgeom::GeoManager::Instance()), m_currLV(nullptr),
m_matphys(nullptr), m_currState(nullptr), m_nextState(nullptr),
m_hist2d(new Hist2D(-500,500,100,-500,500,100))
{
  m_currState = vecgeom::NavigationState::MakeInstance(m_geo.getMaxDepth());
  m_nextState = vecgeom::NavigationState::MakeInstance(m_geo.getMaxDepth());
}

Prompt::NavManager::~NavManager()
{
  m_hist2d->save("promt_first_hist.dat");
  delete m_hist2d;
  delete m_currState;
  delete m_nextState;

}

Prompt::Material *getLogicalVolumePhysics(const vecgeom::LogicalVolume &lv)
{
  return (Prompt::Material *)(lv.GetUserExtensionPtr());
}

void Prompt::NavManager::locateLogicalVolume(const Vector &p)
{
  vecgeom::GlobalLocator::LocateGlobalPoint(m_geo.GetWorld(),
                          {p.x(), p.y(), p.z()}, *m_currState, true);
}

bool Prompt::NavManager::exitWorld()
{
  return m_currState->IsOutside();
}

void Prompt::NavManager::setupVolumePhysics()
{
  //Find next step
  m_currLV = m_currState->Top()->GetLogicalVolume();
  m_matphys = getLogicalVolumePhysics(*m_currLV);
}

size_t Prompt::NavManager::getVolumeID()
{
  return m_currLV->id();
}

std::string Prompt::NavManager::getVolumeName()
{
  return m_currLV->GetName();
}

bool Prompt::NavManager::proprogateInAVolume(Particle &particle, bool verbose )
{
  Vector &p = particle.getPosition();
  Vector &dir = particle.getDirection();

  if (verbose) {
    std::cout << m_currLV->GetName() << ", id " << m_currLV->id() << std::endl;
    std::cout << "initial conditions: pos " << p << " , dir "  << dir  << " ekin " << particle.getEKin() << std::endl;
  }

  if(m_currLV->id()==1)
  {
    auto loc = m_currState->GlobalToLocal({p.x(), p.y(), p.z()});
    m_hist2d->fill(loc[0], loc[1]);
  }

  double stepLength = m_matphys->sampleStepLength(particle.getEKin(), dir);

  vecgeom::Vector3D<Precision> pos{p.x(), p.y(), p.z()};
  vecgeom::Vector3D<Precision> direction{dir.x(), dir.y(), dir.z()};

  //! updates m_nextState to contain information about the next hitting boundary:
  //!   - if a daugher is hit: m_nextState.Top() will be daughter
  //!   - if ray leaves volume: m_nextState.Top() will point to current volume
  //!   - if step limit > step: m_nextState == in_state
  //!   ComputeStep is essentialy equal to ComputeStepAndPropagatedState without the relaction part
  double step = m_currLV->GetNavigator()->ComputeStepAndPropagatedState(pos, direction, stepLength, *m_currState, *m_nextState);
  std::swap(m_currState, m_nextState);

  bool sameVolume = step == stepLength;
  assert(stepLength >= step);
  if (verbose && !sameVolume) { std::cout << "hitDaugherBoundary\n";}

  //Move next step
  const double resolution = 1e-13;
  pos += (step + sameVolume ? 0 : resolution) * direction;
  particle.moveForward(step);

  if (verbose) {
    std::cout << "scattered conditions: pos " << p << " , dir "  << dir  << " ekin " << particle.getEKin() << " step " << step << std::endl << std::endl;
  }

  if(sameVolume)
  {
    //sample the interaction at the location
    double final_ekin(0);
    Vector final_dir;
    m_matphys->sampleFinalState(particle.getEKin(), dir, final_ekin, final_dir);
    particle.changeEKinTo(final_ekin);
    particle.changeDirectionTo(final_dir);
    return true;
  }
  else
    return false;
}
