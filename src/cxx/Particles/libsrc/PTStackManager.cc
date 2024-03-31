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

#include "PTStackManager.hh"


std::ostream& Prompt::operator << (std::ostream &o, const StackManager&s)
{
  o<<"********StackManager*****************\n";

  for(auto it=s.m_stack.begin();it!=s.m_stack.end();++it)
  {
    o<< *it->get() << std::endl;
    o<<"*************************\n";
  }
  return o;
}

Prompt::StackManager::StackManager(): m_unweighted(0)
{

}

void Prompt::StackManager::add(std::unique_ptr<Prompt::Particle> aParticle)
{
  m_stack.emplace_back(std::move(aParticle));
}

void Prompt::StackManager::addSecondary(std::unique_ptr<Prompt::Particle> aParticle)
{
  m_stack.emplace_back(std::move(aParticle));
  m_unweighted++;
}
    
void Prompt::StackManager::scalceSecondary(int lastidx, double factor)
{
  m_stack[m_stack.size()-1-lastidx]->scaleWeight(factor);
  m_unweighted--;
}

void Prompt::StackManager::add(const Prompt::Particle& aparticle, unsigned number)
{
  pt_assert_always(number);
  for(unsigned i=0;i<number;i++)
  {
    auto np = std::make_unique<Particle>(aparticle);
    m_stack.emplace_back(std::move(np));
  }
}


std::unique_ptr<Prompt::Particle> Prompt::StackManager::pop()
{
  if(m_unweighted)
  {
    PROMPT_THROW2(CalcError, "The weight of " << m_unweighted << " particles in the particle stack are not corrected");
  }
  auto p = std::move(m_stack.back());
  m_stack.pop_back();
  return p;
}


bool Prompt::StackManager::empty() const
{
  return m_stack.empty();
}
