#ifndef Prompt_MaterialDecomposer_hh
#define Prompt_MaterialDecomposer_hh

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

#include <string>
#include <memory>
#include <map>

#include "PromptCore.hh"
#include "PTSingleton.hh"

namespace Prompt {

    struct IsotopeComposition {
    int Z;
    int A;
    double frac;
    std::string name;
  };

  class MaterialDecomposer {
  public:
    std::vector<Prompt::IsotopeComposition>& getComposition(int Z);
    void parseCfgStr(const std::string & str);

  private:
  
    friend class Singleton<MaterialDecomposer>;
    MaterialDecomposer();
    ~MaterialDecomposer();

    std::map<int, std::vector<Prompt::IsotopeComposition>> m_comp;

  };
}

#endif
