#include "PTTrackManager.hh"

Prompt::TrackManager::TrackManager()
{

}

Prompt::TrackManager::~TrackManager()
{

}


void Prompt::TrackManager::addTrack(std::unique_ptr<Prompt::Track> &aTrack)
{
  m_tracks.emplace_back(std::move(aTrack));
}
