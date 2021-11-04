#include "PTAnaManager.hh"
#include "PTUtils.hh"
#include "PTScororNeutronSq.hh"

Prompt::AnaManager::AnaManager()
{}

Prompt::AnaManager::~AnaManager()
{}


std::shared_ptr<Prompt::Scoror> Prompt::AnaManager::createScoror(const std::string &definition)
{
  std::vector<std::string> words;
  split(definition, ';', words);
  return std::make_shared<Prompt::ScororNeutronSq>(Vector{0,0,0.}, Vector{0,0,1.}, 12000.*Unit::mm, 0., 30.,300);
}
