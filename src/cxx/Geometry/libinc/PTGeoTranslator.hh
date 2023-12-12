#ifndef Prompt_GeoTranslator_hh
#define Prompt_GeoTranslator_hh

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

#include "PromptCore.hh"
#include <VecGeom/navigation/BVHNavigator.h>

namespace Prompt {

  class GeoTranslator  {
  public:
    GeoTranslator();
    ~GeoTranslator();

    void print() const {
      m_trans.Print();
      std::cout << "\n";
    }

    vecgeom::Transformation3D& getTransformMatrix() { return m_trans; };

    Vector global2Local(const Vector&) const;
    Vector local2Global(const Vector&) const;

    Vector global2Local_direction(const Vector&) const;
    Vector local2Global_direction(const Vector&) const;

  private:
    vecgeom::Transformation3D m_trans;
  };
}

#endif
