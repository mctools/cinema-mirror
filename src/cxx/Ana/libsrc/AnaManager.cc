#include "AnaManager.hh"

Prompt::AnaManager::AnaManager()
{}

Prompt::AnaManager::~AnaManager()
{}

void Prompt::AnaManager::bookScorer(std::string volumeName, AnalysisType type)
{
  m_volNameID.emplace(volumeName, type);
}

void Prompt::AnaManager::initScorer(size_t id, AnalysisType type)
{

}
