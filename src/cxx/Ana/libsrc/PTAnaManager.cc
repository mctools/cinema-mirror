#include "PTAnaManager.hh"
#include "PTUtils.hh"
#include "PTScororNeutronSq.hh"
#include "PTScororPSD.hh"
#include "PTScororVolFlux.hh"

Prompt::AnaManager::AnaManager()
{}

Prompt::AnaManager::~AnaManager()
{}


std::shared_ptr<Prompt::Scoror> Prompt::AnaManager::createScoror(const std::string &cfg, double vol)
{
  auto words = split(cfg, ';');
  std::cout << "Creating scoror with config: ";
  std::cout << cfg << "\n";

  if(words[0]=="NeutronSq")
  {
    //type
    auto samplePos = string2vec(words[2]);
    auto neutronDir = string2vec(words[3]);
    double moderator2SampleDist = std::stod(words[4]);
    double minQ = std::stod(words[5]);
    double maxQ = std::stod(words[6]);
    int numBin = std::stoi(words[7]);
    return std::make_shared<Prompt::ScororNeutronSq>(words[1], samplePos, neutronDir, moderator2SampleDist, minQ, maxQ, numBin);
  }
  else if(words[0]=="PSD")
  {
    return std::make_shared<Prompt::ScororPSD>(words[1], std::stod(words[2]) , std::stod(words[3]) , std::stoi(words[4]) ,
                                          std::stod(words[5]) , std::stod(words[6]) , std::stoi(words[7]) );

  }
  else if(words[0]=="VolFlux")
  {
    std::cout << words[1] << " "
    <<  std::stod(words[2]) << " "
    <<  std::stod(words[3]) << " "
    <<  std::stoi(words[4]) << " "
    <<  std::stoi(words[5]) << " "
    <<  vol << std::endl;
    return std::make_shared<Prompt::ScororVolFlux>(words[1], std::stod(words[2]) ,
                std::stod(words[3]) , std::stoi(words[4]) ,  std::stoi(words[5]),
                vol );

  }
  else
    PROMPT_THROW2(BadInput, "Scoror type " << words[0] << " is not supported. ")
}
