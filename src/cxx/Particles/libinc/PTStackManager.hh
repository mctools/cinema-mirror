#ifndef Prompt_SingletonStackManager_hh
#define Prompt_SingletonStackManager_hh

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

#include "PTSingleton.hh"
#include "PromptCore.hh"
#include "PTParticle.hh"
#include <ostream>

namespace Prompt {

  class StackManager {
  public:
    void add(std::unique_ptr<Particle> aparticle);
    void add(const Particle& aparticle, unsigned number);
    std::unique_ptr<Particle> pop();
    bool empty() const;

    friend std::ostream& operator << (std::ostream &, const StackManager&);


  private:
    friend class Singleton<StackManager>;
    StackManager() = default;
    ~StackManager() = default;
    std::vector<std::unique_ptr<Particle> > m_stack;
  };

  std::ostream& operator << (std::ostream &, const StackManager&);
}

#endif
