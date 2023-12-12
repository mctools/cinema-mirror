#ifndef Prompt_MCPLGun_hh
#define Prompt_MCPLGun_hh

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

#include "PTPrimaryGun.hh"
#include "PTMCPLParticleReader.hh"

namespace Prompt {
  class MCPLGun : public PrimaryGun {
  public:
    MCPLGun(const Particle &aParticle, const std::string &fn)
    : PrimaryGun(aParticle), m_mcplread(std::make_unique<MCPLParticleReader>(fn)) {  };
    virtual ~MCPLGun() { }; 

    virtual std::unique_ptr<Particle> generate() override
    {
      m_mcplread->readOneParticle();
      return PrimaryGun::generate();
    }

    virtual void sampleEnergy(double &ekin) override
    {
      ekin=m_mcplread->getEnergy();
    }

    virtual void samplePosDir(Vector &pos, Vector &dir) override
    {
      m_mcplread->getPosition(pos);
      m_mcplread->getDirection(dir);
    }

    virtual double getParticleWeight() override
    { 
      return m_mcplread->getWeight();
    }

    virtual double getTime() override
    { 
      return m_mcplread->getTime();
    }

    private:
      std::unique_ptr <MCPLParticleReader>  m_mcplread;
  };
}


#endif
