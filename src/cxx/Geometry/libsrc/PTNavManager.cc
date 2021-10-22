
#include "PTNavManager.hh"
#include <VecGeom/navigation/BVHNavigator.h>
#include <VecGeom/navigation/NewSimpleNavigator.h>

Prompt::NavManager::NavManager()
:m_geo(vecgeom::GeoManager::Instance()), m_currVolume(nullptr),
m_matphys(nullptr), m_currState(nullptr), m_nextState(nullptr)
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
  m_currState = vecgeom::NavigationState::MakeInstance(m_geo.getMaxDepth());
  m_nextState = vecgeom::NavigationState::MakeInstance(m_geo.getMaxDepth());

  vecgeom::GlobalLocator::LocateGlobalPoint(m_geo.GetWorld(),
                          {p.x(), p.y(), p.z()}, *m_currState, true);

  m_nextState->Clear();
}

bool Prompt::NavManager::proprogate(Particle &particle, bool verbose )
{
  Vector &p = particle.getPosition();
  Vector &dir = particle.getDirection();
  locateLogicalVolume(p);

  while (!m_currState->IsOutside()) {
    //Find next step
    m_currVolume = m_currState->Top()->GetLogicalVolume();
    m_matphys = getLogicalVolumePhysics(*m_currVolume);

    if (verbose) {
      std::cout << m_currVolume->GetName() << std::endl;
      std::cout << "initial conditions: pos " << p << " , dir "  << dir  << " ekin " << particle.getEKin() << std::endl;
    }

    double stepLength = m_matphys->sampleStepLength(particle.getEKin(), dir);

    vecgeom::Vector3D<Precision> pos{p.x(), p.y(), p.z()};
    vecgeom::Vector3D<Precision> direction{dir.x(), dir.y(), dir.z()};

    //! updates m_nextState to contain information about the next hitting boundary:
    //!   - if a daugher is hit: m_nextState.Top() will be daughter
    //!   - if ray leaves volume: m_nextState.Top() will point to current volume
    //!   - if step limit > step: m_nextState == in_state
    //!   ComputeStep is essentialy equal to ComputeStepAndPropagatedState without the relaction part
    double step = m_currVolume->GetNavigator()->ComputeStepAndPropagatedState(pos, direction, stepLength, *m_currState, *m_nextState);

    bool sameVolume = step == stepLength;
    if (verbose && !sameVolume) { std::cout << "hitDaugherBoundary\n";}

    //Move next step
    const double resolution = 1e-13;
    pos += (step + sameVolume ? 0 : resolution) * direction;
    particle.moveForward(step);

    if(sameVolume)
    {
      //sample the interaction at the location
      double final_ekin(0);
      Vector final_dir;
      m_matphys->sampleFinalState(particle.getEKin(), dir, final_ekin, final_dir);
      particle.changeEKinTo(final_ekin);
      particle.changeDirectionTo(final_dir);
    }
    std::swap(m_currState, m_nextState);

    if (verbose) {
      std::cout << "scattered conditions: pos " << p << " , dir "  << dir  << " ekin " << particle.getEKin() << " step " << step << std::endl << std::endl;
    }
  }

  return false;
}
