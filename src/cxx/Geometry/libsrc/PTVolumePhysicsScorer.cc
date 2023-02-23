#include "PTVolumePhysicsScorer.hh"

void Prompt::VolumePhysicsScorer::sortScorers()
{
  entry_scorers.clear();
  propagate_scorers.clear();
  exit_scorers.clear();
  surface_scorers.clear();
  absorb_scorers.clear();
  
  if(scorers.size())
    std::cout << "Sorting "<< scorers.size() << " scorers \n\n";

  for(auto &v : scorers)
  {
    auto type = v->getType();
    if(type==Scorer::ScorerType::ENTRY)
    {
      entry_scorers.push_back(v);
      std::cout << "Added ENTRY type scorer: " << v->getName() << std::endl;
    }
    else if(type==Scorer::ScorerType::PROPAGATE)
    {
      propagate_scorers.push_back(v);
      std::cout << "Added PROPAGATE type scorer: " << v->getName() << std::endl;
    }
    else if(type==Scorer::ScorerType::EXIT)
    {
      exit_scorers.push_back(v);
      std::cout << "Added EXIT type scorer: " << v->getName() << std::endl;
    }
    else if(type==Scorer::ScorerType::SURFACE)
    {
      surface_scorers.push_back(v);
      std::cout << "Added SURFACE type scorer: " << v->getName() << std::endl;
    }
    else if(type==Scorer::ScorerType::ABSORB)
    {
      absorb_scorers.push_back(v);
      std::cout << "Added ABSORB type scorer: " << v->getName() << std::endl;
    }
    else if(type==Scorer::ScorerType::ENTRY2EXIT)
    {
      entry_scorers.push_back(v);
      propagate_scorers.push_back(v);
      exit_scorers.push_back(v);
      std::cout << "Added ENTRY2EXIT type scorer: " << v->getName() << std::endl;
    }
    else
      PROMPT_THROW2(BadInput, "unknown scorer type " << static_cast<int>(type) );
  }
  std::cout << "\n";
}
