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


#include "PromptCore.hh"
// #include "PTPython.hh"


// PyObject* get(PyObject* &module, const char* name) 
// {
//     PyObject* ret = PyObject_GetAttrString(module, name);
//     if(!ret)
//         PROMPT_THROW2(BadInput, "Can not get attribute " << name );
//     return ret;
// }

// PyObject *pt_call_python_method(PyObject *obj, const char* method)
// {
//     // https://www.geeksforgeeks.org/releasing-gil-and-mixing-threads-from-c-and-python/
//     // For any C involving Python objects or the Python C API, GIL needs to be acquired and released first. 
//     // This can be performed using PyGILState_Ensure() and PyGILState_Release() as in the code given below.
//     // Every call to PyGILState_Ensure() must have a matching call to PyGILState_Release().

//     PyGILState_STATE state = PyGILState_Ensure();
//     if(!obj)
//         PROMPT_THROW(BadInput, "Input PyObject is not valid")
//     auto ret = PyObject_CallMethod(obj, method,"()"); 
//     PyGILState_Release(state);
//     Py_INCREF(ret);   //  remember to Py_DECREF after use
//     return ret;
// }

