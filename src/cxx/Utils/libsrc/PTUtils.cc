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
#include <algorithm>

std::vector<std::string> Prompt::split(const std::string& text, char delimiter)
{
  std::vector<std::string> words;
  std::stringstream sstream(text);
  std::string word;
  while (std::getline(sstream, word, delimiter))
  {
    word.erase(std::remove_if(word.begin(), word.end(),
                            [](char c) {
                                return (c == ' ' || c == '\n' || c == '\r' ||
                                        c == '\t' || c == '\v' || c == '\f');
                            }),
                            word.end());
    words.push_back(word);
  }


  return words;
}

Prompt::Vector Prompt::string2vec(const std::string& text, char delimiter)
{
  auto subs = split(text, delimiter);
  if(subs.size()!=3)
    PROMPT_THROW2(BadInput, "string2vec failed to create a vector from the input string " << text);
  return Vector{ptstod(subs[0]), ptstod(subs[1]), ptstod(subs[2]) };
}

int Prompt::ptstoi(const std::string& text)
{
  try
  {
    return std::stoi(text);
  }
  catch(...)
  {
    // std::invalid_argument, std::out_of_range
    PROMPT_THROW2(BadInput, "ptstod filed to a int from the input string " << text);
  }
}


double Prompt::ptstod(const std::string& text)
{
  try
  {
    return std::stod(text);
  }
  catch(...)
  {
    // std::invalid_argument, std::out_of_range
    PROMPT_THROW2(BadInput, "ptstod filed to a double from the input string " << text);
  }
}
