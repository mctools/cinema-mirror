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

#include "PTUtils.hh"

std::vector<std::string> Prompt::split(const std::string& text, char delimiter)
{
  std::vector<std::string> words;
  std::stringstream sstream(text);
  std::string word;
  while (std::getline(sstream, word, delimiter))
      words.push_back(word);

  return words;
}

Prompt::Vector Prompt::string2vec(const std::string& text, char delimiter)
{
  auto subs = split(text, delimiter);
  if(subs.size()!=3)
    PROMPT_THROW2(BadInput, "string2vec " << text);
  return Vector{std::stod(subs[0]), std::stod(subs[1]),std::stod(subs[2]) };
}
