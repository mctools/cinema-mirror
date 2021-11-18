#ifndef Prompt_GeoManager_hh
#define Prompt_GeoManager_hh

#include <string>
#include <map>
#include <unordered_map>
#include "PromptCore.hh"
#include "PTMaterialPhysics.hh"
#include "PTSingleton.hh"
#include "PTScoror.hh"
#include "PTPrimaryGun.hh"

namespace Prompt {

  struct VolumePhysicsScoror { // to attached to a volume
    std::shared_ptr<MaterialPhysics> physics;
    std::vector< std::shared_ptr<Scoror> >  scorors; /*scoror name, scoror*/

    std::vector< std::shared_ptr<Scoror> >  entry_scorors;
    std::vector< std::shared_ptr<Scoror> >  propagate_scorors;
    std::vector< std::shared_ptr<Scoror> >  exit_scorors;

    void sortScorors()
    {
      entry_scorors.clear();
      propagate_scorors.clear();
      exit_scorors.clear();

      for(auto &v : scorors)
      {
        auto type = v->getType();
        if(type==ScororType::ENTRY)
        {
          entry_scorors.push_back(v);
        }
        else if(type==ScororType::PROPAGATE)
        {
          propagate_scorors.push_back(v);
        }
        else if(type==ScororType::EXIT)
        {
          exit_scorors.push_back(v);
        }
        else
          PROMPT_THROW(BadInput, "unknown scoror type")
      }
    }
  };
  using VPSMap = std::unordered_map<size_t, std::shared_ptr<VolumePhysicsScoror>>;

  class GeoManager  {
  public:
    void loadFile(const std::string &loadFile);
    std::shared_ptr<MaterialPhysics> getMaterialPhysics(const std::string &name);
    std::shared_ptr<Scoror> getScoror(const std::string &name);
    size_t numMaterialPhysics() {return m_globelPhysics.size();}
    size_t numScoror() {return m_globelScorors.size();}

    VPSMap::const_iterator getVolumePhysicsScoror(size_t id)
    {
      auto it = m_volphyscoror.find(id);
      assert(it!=m_volphyscoror.end());
      return it;
    }

    std::shared_ptr<PrimaryGun> m_gun;


  private:
    friend class Singleton<GeoManager>;

    GeoManager();
    ~GeoManager();

    // the name is unique
    std::map<std::string /*material name*/, std::shared_ptr<MaterialPhysics> > m_globelPhysics;
    std::map<std::string /*scoror name*/, std::shared_ptr<Scoror> >  m_globelScorors;

    //the place to manage the life time of MaterialPhysics scorors
    std::unordered_map<size_t, std::shared_ptr<VolumePhysicsScoror>> m_volphyscoror;
  };
}

#endif
