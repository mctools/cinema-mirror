#include <string>

#include <VecGeom/base/Config.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/gdml/Frontend.h>
#include <VecGeom/gdml/Middleware.h>
#include <VecGeom/navigation/BVHNavigator.h>
#include <VecGeom/navigation/NewSimpleNavigator.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <random>

#include <err.h>
#include <getopt.h>
#include <libgen.h>

using namespace vecgeom;

static std::random_device rd;
static std::default_random_engine rng;
static std::uniform_real_distribution<float> dist(0.0f, 1.0f);

bool nearly_equal(double x, double y)
{
  if (x == y)
    return true;
  else if (x * y == 0.0)
    return abs(x - y) < DBL_EPSILON * DBL_EPSILON;
  else
    return abs(x - y) < (abs(x) + abs(y)) * DBL_EPSILON;
}


double uniform(double a, double b)
{
  return a + (b - a) * dist(rng);
}


Vector3D<Precision> random_unit_vector()
{
  Precision z = uniform(-1.0f, 1.0f);
  Precision r = sqrt(1.0f - z * z);
  Precision t = uniform(0.0f, 6.2831853f);
  return {r * cos(t), r * sin(t), z};
}


bool navigate(Vector3D<Precision> &p, Vector3D<Precision> dir, double stepLength, bool verbose = true)
{
  auto &geoManager      = GeoManager::Instance();
  NavigationState *curr = NavigationState::MakeInstance(geoManager.getMaxDepth());
  NavigationState *next = NavigationState::MakeInstance(geoManager.getMaxDepth());

  VNavigator const &ref_navigator = *NewSimpleNavigator<>::Instance();

  GlobalLocator::LocateGlobalPoint(geoManager.GetWorld(), p, *curr, true);

  LogicalVolume const *curr_volume = curr->Top()->GetLogicalVolume();

  if (verbose) {
    printf("initial conditions:\n\n\t   volume: %s"
           "\n\t position: [ % .8f, % .8f, % .8f ]\n\tdirection: [ % .8f, % .8f, % .8f ]\n\n",
           curr_volume->GetLabel().c_str(), p.x(), p.y(), p.z(), dir.x(), dir.y(), dir.z());
    printf("%6s%25s%-25s%15s%15s%8s\n\n", "step", "", "position", "step length", "reference", "volume");
  }

  size_t steps = 0;
  // while (!curr->IsOutside())
  {
    curr_volume = curr->Top()->GetLogicalVolume();
    double ref_step = ref_navigator.ComputeStepAndPropagatedState(p, dir, stepLength, *curr, *next);
    double step     = curr_volume->GetNavigator()->ComputeStepAndPropagatedState(p, dir, stepLength, *curr, *next);
    if (!nearly_equal(step, ref_step)) return false;

    p = p + step * dir;

    std::swap(curr, next);

    if (verbose)
      printf("%6lu [ % 14.8f, % 14.8f, % 14.8f ] % 14.8f % 14.8f %s\n", ++steps, p.x(), p.y(), p.z(), step, ref_step,
             curr_volume->GetLabel().c_str());
  }

  if (verbose) printf("\n");

  return !curr->IsOutside();
}


int main(int argc, char** argv)
{
#ifndef VECGEOM_GDML
  std::cout << "### VecGeom must be compiled with GDML support to run this.\n";
  return 1;
#endif

#ifndef VECGEOM_USE_NAVINDEX
    std::cout << "### VecGeom must be compiled with USE_NAVINDEX support to run this.\n";
    return 2;
#endif

  std::string gdml_name=(argv[1]);

  //access material
  vgdml::Parser p;
  auto const loadedMiddleware = p.Load(gdml_name.c_str(), false, 1); // mm unit is 1
  std::cout << "Geometry loaded with result: \"" << (loadedMiddleware ? "true" : "false") << "\"" << std::endl;
  if (!loadedMiddleware) return 1;

  auto const &aMiddleware = *loadedMiddleware;
  auto volumeMatMap   = aMiddleware.GetVolumeMatMap();

  //initialise navigation
  auto &geoManager = vecgeom::GeoManager::Instance();
  auto navigator = vecgeom::BVHNavigator<>::Instance();

  for (auto &item : geoManager.GetLogicalVolumesMap())
  {
    auto &volume   = *item.second;
    auto nchildren = volume.GetDaughters().size();
    volume.SetNavigator(nchildren > 0 ? navigator : NewSimpleNavigator<>::Instance());

    const vgdml::Material& mat = volumeMatMap[volume.id()];

    std::cout << "volume name " << volume.GetName() << " (id = " << volume.id() << "): material name " << mat.name << std::endl;
    if (mat.attributes.size()) std::cout << "  attributes:\n";
    for (const auto &attv : mat.attributes)
      std::cout << "    " << attv.first << ": " << attv.second << std::endl;
    if (mat.fractions.size()) std::cout << "  fractions:\n";
    for (const auto &attv : mat.fractions)
      std::cout << "    " << attv.first << ": " << attv.second << std::endl;
    if (mat.components.size()) std::cout << "  components:\n";
    for (const auto &attv : mat.components)
      std::cout << "    " << attv.first << ": " << attv.second << std::endl;
  }

  vecgeom::BVHManager::Init();
  size_t seed=10101;
  rng.seed(seed ? seed : seed = rd());
  Vector3D<Precision> pos(0.0, 0.0, 0.0);
  Vector3D<Precision> dir ;

  for (unsigned long i = 0; i < 1000; ++i)
  {
    dir = random_unit_vector();
    if (navigate(pos, dir, dist(rng)*100, 0))
    {
      std::cout << "step " << i << ", pos" << pos << std::endl;
      continue;
    }
    else
    {
      std::cout << "exit step " << i << ", pos" << pos << std::endl;
      break;
    }
  }
}
