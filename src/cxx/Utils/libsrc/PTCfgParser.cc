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
#include <cxxabi.h> //__cxa_demangle

Prompt::CfgParser::CfgParser()
{

}

std::string Prompt::CfgParser::getTypeName(const std::type_info& ti)
{
  // see https://panthema.net/2008/0901-stacktrace-demangled/cxa_demangle.html and
  // https://gcc.gnu.org/onlinedocs/libstdc++/manual/ext_demangling.html

  std::string tname (abi::__cxa_demangle(ti.name(), nullptr, nullptr, nullptr));

  std::string substr = "Prompt::";
  std::string::size_type i = tname.find(substr);
  if (i != std::string::npos)
     tname.erase(i, substr.length());

  return std::move(tname);
}

Prompt::CfgParser::ScorerCfg Prompt::CfgParser::getScorerCfg(const std::string& cfgstr)
{
  auto strvec = split(cfgstr, ';');
  ScorerCfg cfg;
  cfg.name=strvec[0];
  for(const auto &s: strvec)
  {
    auto p = split(s, '=');
    cfg.parameters[p[0]] = p[1];
    // std::cout << s << std::endl;
  }
  return std::move(cfg);
}
