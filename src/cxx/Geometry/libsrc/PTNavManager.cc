
#include "PTNavManager.hh"
#include <VecGeom/navigation/BVHNavigator.h>
#include <VecGeom/navigation/NewSimpleNavigator.h>

Prompt::NavManager::NavManager()
:m_geo(vecgeom::GeoManager::Instance()), m_currVolume(nullptr), m_matphys(nullptr)
{
}

Prompt::NavManager::~NavManager()
{}

Prompt::Material *getLogicalVolumePhysics(const vecgeom::LogicalVolume &lv)
{
  return (Prompt::Material *)(lv.GetUserExtensionPtr());
}

void Prompt::NavManager::locateLogicalVolume(const Vector &p)
{
  vecgeom::GlobalLocator::LocateGlobalPoint(m_geo.GetWorld(), {p.x(), p.y(), p.z()}, *m_currState, true);
  if(!m_currState)
    printf("m_currState is nullptr\n");

  m_currVolume = const_cast<vecgeom::LogicalVolume*> (m_currState->Top()->GetLogicalVolume());
  m_matphys = getLogicalVolumePhysics(*m_currVolume);
}

bool Prompt::NavManager::proprogate(Particle &particle, bool verbose )
{
  m_currState = vecgeom::NavigationState::MakeInstance(m_geo.getMaxDepth());
  m_nextState = vecgeom::NavigationState::MakeInstance(m_geo.getMaxDepth());

  Vector &p = particle.getPosition();
  Vector &dir = particle.getDirection();

  locateLogicalVolume(p);

  if (verbose) {
    std::cout << "initial conditions: pos " << p << " , dir "  << dir  << " ekin " << particle.getEKin()<< std::endl;
  }

  double stepLength = m_matphys->sampleStepLength(particle.getEKin(), dir);

  vecgeom::Vector3D<Precision> pos{p.x(), p.y(), p.z()};
  vecgeom::Vector3D<Precision> direction{dir.x(), dir.y(), dir.z()};

  double step = m_currVolume->GetNavigator()->ComputeStepAndPropagatedState(pos, direction, stepLength, *m_currState, *m_nextState);
  pos += step * direction;
  particle.moveForward(step);

  double final_ekin(0);
  Vector final_dir;
  m_matphys->sampleFinalState(particle.getEKin(), dir, final_ekin, final_dir);
  particle.changeEKinTo(final_ekin);
  particle.changeDirectionTo(final_dir);

  std::swap(m_currState, m_nextState);

  if (verbose) {
    std::cout << "scattered conditions: pos " << p << " , dir "  << dir  << " ekin " << particle.getEKin()<< std::endl << std::endl;
  }
  return !m_currState->IsOutside();
}
