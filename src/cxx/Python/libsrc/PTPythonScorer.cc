////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//  This file is part of Prompt (see https://gitlab.com/xxcai1/Prompt)        //
//                                                                            //
//  Copyright 2021-2024 Prompt developers                                     //
//                                                                            //
//  Licensed under the Apache License, Version 2.0 (the "License");           //
//  you may not use this file except in compliance with the License.          //
//  You may obtain a copy of the License at                                   //
//                                                                            //
//      http://www.apache.org/licenses/LICENSE-2.0                            //
//                                                                            //
//  Unless required by applicable law or agreed to in writing, software       //
//  distributed under the License is distributed on an "AS IS" BASIS,         //
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  //
//  See the License for the specific language governing permissions and       //
//  limitations under the License.                                            //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include "PTPythonScorer.hh"
#include "PTScorerDeposition.hh"
#include "PTScorerESpectrum.hh"
#include "PTScorerTOF.hh"
#include "PTScorerWlSpectrum.hh"
#include "PTScorerVolFluence.hh"
#include "PTScorerMultiScat.hh"
#include "PTScorerDirectSqw.hh"
#include "PTScorer2D.hh"
namespace pt = Prompt;


//ScorerDeposition
void* pt_ScorerDeposition_new(const char* name, double xmin, double xmax, unsigned nbins, unsigned pdg, int type, bool linear, int groupid)
{
  pt::ScorerDeposition::ScorerType t = static_cast<pt::ScorerDeposition::ScorerType>(type); 
  return static_cast<void *>(new pt::ScorerDeposition(name, xmin, xmax, nbins, pdg, t, linear, groupid));
}

void* pt_ScorerESpectrum_new(const char* name, bool scoreTransfer, double xmin, double xmax, unsigned nxbins, unsigned int pdg, int type, int groupid, bool linear)
{
  pt::ScorerESpectrum::ScorerType t = static_cast<pt::ScorerESpectrum::ScorerType>(type); 
  return static_cast<void *>(new pt::ScorerESpectrum(name, scoreTransfer, xmin, xmax, nxbins, pdg, t, groupid, linear));
}

void* pt_ScorerTOF_new(const char* name, double xmin, double xmax, unsigned nxbins, unsigned int pdg, int type, int groupid)
{
  pt::ScorerTOF::ScorerType t = static_cast<pt::ScorerTOF::ScorerType>(type); 
  return static_cast<void *>(new pt::ScorerTOF(name, xmin, xmax, nxbins, pdg, t, groupid));
}

void* pt_ScorerWlSpectrum_new(const char* name, double xmin, double xmax, unsigned nxbins, unsigned int pdg, int type, int groupid)
{
  pt::ScorerWlSpectrum::ScorerType t = static_cast<pt::ScorerWlSpectrum::ScorerType>(type); 
  return static_cast<void *>(new pt::ScorerWlSpectrum(name, xmin, xmax, nxbins, pdg, t, groupid));
}

void* pt_ScorerVolFluence_new(const char* name, double xmin, double xmax, unsigned nbins, double volme, unsigned pdg, int type, bool linear, int groupid)
{
  pt::ScorerVolFluence::ScorerType t = static_cast<pt::ScorerVolFluence::ScorerType>(type); 
  return static_cast<void *>(new pt::ScorerVolFluence(name, xmin, xmax, nbins, volme, pdg, t, linear, groupid));
}

void* pt_ScorerMultiScat_new(const char* name, double xmin, double xmax, unsigned nbins, unsigned pdg, int type, int groupid)
{
  pt::ScorerMultiScat::ScorerType t = static_cast<pt::ScorerMultiScat::ScorerType>(type); 
  return static_cast<void *>(new pt::ScorerMultiScat(name, xmin, xmax, nbins, pdg, t, true, groupid));
}

void* pt_ScorerDirectSqw_new(const char* name, double qmin, double qmax, unsigned xbin, 
double ekinmin, double ekinmax, unsigned nybins, unsigned int pdg, int group_id,
double mod_smp_dist, double mean_ekin,
double mean_incident_dir_x, double mean_incident_dir_y, double mean_incident_dir_z,
double sample_position_x, double sample_position_y, double sample_position_z, int type)
{
  auto t = static_cast<pt::Scorer::ScorerType>(type); 
  pt::Vector mean_incident_dir(mean_incident_dir_x, mean_incident_dir_y, mean_incident_dir_z);
  pt::Vector sample_position(sample_position_x, sample_position_y, sample_position_z);

  return static_cast<void *>(new pt::ScorerDirectSqw(name, qmin, qmax, xbin, ekinmin,  
                                        ekinmax, nybins, pdg, group_id, mod_smp_dist, mean_ekin, 
                                        mean_incident_dir, sample_position, t));
}

void pt_addMultiScatter1D(void* scatter, void* espScorer, int scatNumReq=0)
{
  auto esp = static_cast<Prompt::Scorer1D *>(espScorer);
  const auto scat = static_cast<Prompt::ScorerMultiScat *>(scatter);
  esp->addMultiScatter(scat, scatNumReq);
}


void pt_addMultiScatter2D(void* scatter, void* espScorer, int scatNumReq=0)
{
  auto esp = static_cast<Prompt::Scorer2D *>(espScorer);
  const auto scat = static_cast<Prompt::ScorerMultiScat *>(scatter);
  esp->addMultiScatter(scat, scatNumReq);
}
