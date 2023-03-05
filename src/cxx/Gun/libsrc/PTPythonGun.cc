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

#include "PTPythonGun.hh"
#include "PTNeutron.hh"
#include "PTPython.hh"

Prompt::PythonGun::PythonGun(PyObject *obj)
    : PrimaryGun(Neutron()),  m_pyobj(obj)
{  
    Py_INCREF(m_pyobj);
}

Prompt::PythonGun::~PythonGun()
{ 
    Py_DECREF(m_pyobj); 
}

std::unique_ptr<Prompt::Particle> Prompt::PythonGun::generate()
{
    auto ret = pt_call_python_method(m_pyobj, "generate");
    if (! PyArg_ParseTuple(ret, "ddddddddd", &m_ekin, &m_weight, &m_time,
            &m_pos.x(), &m_pos.y(), &m_pos.z(),
            &m_dir.x(), &m_dir.y(), &m_dir.z()))
    {
        PROMPT_THROW(BadInput, "PyArg_ParseTuple failed\n");
    }
    Py_DECREF(ret);

    m_ekin0=m_ekin;
    m_eventid++;
    m_alive = true;
    auto p = std::make_unique<Particle>(*this);
    // std::cout << *p << std::endl;
    return std::move(p);
}

void* pt_PythonGun_new(PyObject *pyobj)
{
    return static_cast<void *> (new Prompt::PythonGun(pyobj)) ;
}

void pt_PythonGun_delete(void *obj)
{
    delete static_cast<Prompt::PythonGun *> (obj);
}

void pt_PythonGun_generate(void *obj)
{
    static_cast<Prompt::PythonGun *> (obj)->generate();
}
