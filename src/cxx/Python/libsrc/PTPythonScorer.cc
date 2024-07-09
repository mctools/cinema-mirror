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

namespace pt = Prompt;


//ScorerDeposition
void* pt_ScorerDeposition_new(const char* name, double xmin, double xmax, unsigned nbins, unsigned pdg, int type, bool linear)
{
  pt::ScorerDeposition::ScorerType t = static_cast<pt::ScorerDeposition::ScorerType>(type); 
  return static_cast<void *>(new pt::ScorerDeposition(name, xmin, xmax, nbins, pdg, t, linear));
}

void* pt_ScorerESpectrum_new(const char* name, bool scoreTransfer, double xmin, double xmax, unsigned nxbins, unsigned int pdg, int type, int groupid)
{
  pt::ScorerESpectrum::ScorerType t = static_cast<pt::ScorerESpectrum::ScorerType>(type); 
  return static_cast<void *>(new pt::ScorerESpectrum(name, scoreTransfer, xmin, xmax, nxbins, pdg, t, groupid));
}

void* pt_ScorerTOF_new(const char* name, double xmin, double xmax, unsigned nxbins, unsigned int pdg, int type, int groupid)
{
  pt::ScorerTOF::ScorerType t = static_cast<pt::ScorerTOF::ScorerType>(type); 
  return static_cast<void *>(new pt::ScorerTOF(name, xmin, xmax, nxbins, pdg, t, groupid));
}

void* pt_ScorerWlSpectrum_new(const char* name, double xmin, double xmax, unsigned nxbins, unsigned int pdg, int type)
{
  pt::ScorerWlSpectrum::ScorerType t = static_cast<pt::ScorerWlSpectrum::ScorerType>(type); 
  return static_cast<void *>(new pt::ScorerWlSpectrum(name, xmin, xmax, nxbins, pdg, t));
}

void* pt_ScorerVolFluence_new(const char* name, double xmin, double xmax, unsigned nbins, double volme, unsigned pdg, int type, bool linear)
{
  pt::ScorerVolFluence::ScorerType t = static_cast<pt::ScorerVolFluence::ScorerType>(type); 
  return static_cast<void *>(new pt::ScorerVolFluence(name, xmin, xmax, nbins, volme, pdg, t, linear));
}