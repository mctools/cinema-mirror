#include "PTAnaManager.hh"
#include "PTUtils.hh"
#include "PTScororNeutronSq.hh"

Prompt::AnaManager::AnaManager()
{}

Prompt::AnaManager::~AnaManager()
{}


std::shared_ptr<Prompt::Scoror> Prompt::AnaManager::createScoror(const std::string &cfg)
{
  auto words = split(cfg, ';');
  std::cout << "Creating scoror with config: ";

  if(words[0]!="NeutronSq")
    PROMPT_THROW2(BadInput, "Scoror type " << words[0] << " is not supported. ")

  //type
  auto samplePos = string2vec(words[1]);
  auto neutronDir = string2vec(words[2]);
  double moderator2SampleDist = std::stod(words[3]);
  double minQ = std::stod(words[4]);
  double maxQ = std::stod(words[5]);
  int numBin = std::stoi(words[6]);

  std::cout << "Scoror config is " << cfg << ".\n"
        << "Parsed value " << samplePos <<  neutronDir
        << moderator2SampleDist << " "
        << minQ << " "
        << maxQ << " "
        << numBin << std::endl;

  return std::make_shared<Prompt::ScororNeutronSq>(samplePos, neutronDir, moderator2SampleDist, minQ, maxQ, numBin);
}
