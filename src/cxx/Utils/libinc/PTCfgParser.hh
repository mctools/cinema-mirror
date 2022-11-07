#ifndef Prompt_CfgParser_hh
#define Prompt_CfgParser_hh

////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//  This file is part of Prompt (see https://gitlab.com/xxcai1/Prompt)        //
//                                                                            //
//  Copyright 2021-2022 Prompt developers                                     //
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
#include <map>
#include "PTSingleton.hh"
#include <iostream>

namespace Prompt {
  class CfgParser {
  public:
    struct ScorerCfg {
      std::string name;
      std::map<std::string, std::string> parameters;
      void print();
      std::string find(const std::string &key, bool force=false);
      bool contains(const std::string &key)
      {
        return find(key).empty() ? false : true;
      }
      size_t size()
      {
        return parameters.size();
      }


      // // if key is not found, the default value will be used
      // template <typename T>
      // T getIfExist(const std::string &key, T &defaultv, int &parCount)
      // {
      //   T value = defaultv;
      //   std::string thresholdInStr = cfg.find(key);
      //   if(thresholdInStr.empty())
      //     parCount--;
      //   else
      //   {
      //     value = ptstod(thresholdInStr);
      //   }
      //   return T;
      // }
    };
  public:
    ScorerCfg parse(const std::string& cfgstr);
  private:
    friend class Singleton<CfgParser>;
    CfgParser();
    ~CfgParser() = default;
  };
}
#endif
