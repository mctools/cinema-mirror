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

#include "PTCfgParser.hh"
#include "PTUtils.hh" //split

void Prompt::CfgParser::ScorerCfg::print()
{
  std::cout << "ScorerCfg " << name << " of size " << size() << ":\n";
  for(auto it = parameters.begin(); it!=parameters.end();++it)
  {
    std::cout << "  [" << it->first << " = "
                << it->second << "]\n";
  }
}

std::string Prompt::CfgParser::ScorerCfg::find(const std::string &key, bool force)
{
  auto  it = parameters.find(key);
  std::string v =  it == parameters.end() ? "" : it->second;
  if(v.empty() && force)
    PROMPT_THROW2(BadInput, "cfg is missing the key \""<< key << "\"" );
  return v;
}

Prompt::CfgParser::CfgParser()
{
}

Prompt::CfgParser::ScorerCfg Prompt::CfgParser::getScorerCfg(const std::string& cfgstr)
{
  auto strvec = split(cfgstr, ';');
  ScorerCfg cfg;
  cfg.name=strvec[0];
  for(const auto &s: strvec)
  {
    auto p = split(s, '=');
    if(p.size()!=2)
        PROMPT_THROW2(BadInput, " cfg section \""<< s << "\" is ill-defined" );
    cfg.parameters[p[0]] = p[1];
    // std::cout << s << std::endl;
  }
  return std::move(cfg);
}
