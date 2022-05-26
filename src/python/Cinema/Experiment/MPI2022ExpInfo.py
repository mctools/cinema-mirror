#!/usr/bin/env python3

################################################################################
##                                                                            ##
##  This file is part of Prompt (see https://gitlab.com/xxcai1/Prompt)        ##
##                                                                            ##
##  Copyright 2021-2022 Prompt developers                                     ##
##                                                                            ##
##  Licensed under the Apache License, Version 2.0 (the "License");           ##
##  you may not use this file except in compliance with the License.          ##
##  You may obtain a copy of the License at                                   ##
##                                                                            ##
##      http://www.apache.org/licenses/LICENSE-2.0                            ##
##                                                                            ##
##  Unless required by applicable law or agreed to in writing, software       ##
##  distributed under the License is distributed on an "AS IS" BASIS,         ##
##  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  ##
##  See the License for the specific language governing permissions and       ##
##  limitations under the License.                                            ##
##                                                                            ##
################################################################################

import json
from Cinema.Interface.Utils import findData

MPIRuns={}

MPIRuns['D2O_NoAu'] = {'run': 'run17672', 'volume(ml)': 0.9, 'height_sample(mm)': 14.3376, 'dmin(mm)': 8.94, 'dmax(mm)': 9.53}
MPIRuns['H2O_NoAu'] = {'run': 'run17673', 'volume(ml)': 0.3, 'height_sample(mm)': 4.7792, 'dmin(mm)': 8.94, 'dmax(mm)': 9.53}
MPIRuns['NaH2PO4_D2O_NoAu'] = {'run': 'run17674', 'volume(ml)': 0.9, 'height_sample(mm)': 14.2419, 'dmin(mm)': 8.97, 'dmax(mm)': 9.53}
MPIRuns['NaH2PO4_H2O_NoAu'] = {'run': 'run17675', 'volume(ml)': 0.3, 'height_sample(mm)': 4.7367, 'dmin(mm)': 8.98, 'dmax(mm)': 9.53}
MPIRuns['CuSO4_D2O_NoAu'] = {'run': 'run17676', 'volume(ml)': 0.9, 'height_sample(mm)': 14.3056, 'dmin(mm)': 8.95, 'dmax(mm)': 9.54}
MPIRuns['CuSO4_H2O_NoAu'] = {'run': 'run17677', 'volume(ml)': 0.3, 'height_sample(mm)': 4.7899, 'dmin(mm)': 8.93, 'dmax(mm)': 9.53}

MPIRuns['D2O_Au'] = {'run': 'run17678', 'volume(ml)': 0.9, 'height_sample(mm)': 14.3376, 'dmin(mm)': 8.94, 'dmax(mm)': 9.53}
MPIRuns['H2O_Au'] = {'run': 'run17679', 'volume(ml)': 0.3, 'height_sample(mm)': 4.7792, 'dmin(mm)': 8.94, 'dmax(mm)': 9.53}
MPIRuns['NaH2PO4_D2O_Au'] = {'run': 'run17680', 'volume(ml)': 0.9, 'height_sample(mm)': 14.2419, 'dmin(mm)': 8.97, 'dmax(mm)': 9.53}
MPIRuns['NaH2PO4_H2O_Au'] = {'run': 'run17681', 'volume(ml)': 0.3, 'height_sample(mm)': 4.7367, 'dmin(mm)': 8.98, 'dmax(mm)': 9.53}
MPIRuns['CuSO4_D2O_Au'] = {'run': 'run17682', 'volume(ml)': 0.9, 'height_sample(mm)': 14.3056, 'dmin(mm)': 8.95, 'dmax(mm)': 9.54}
MPIRuns['CuSO4_H2O_Au'] = {'run': 'run17683', 'volume(ml)': 0.3, 'height_sample(mm)': 4.7899, 'dmin(mm)': 8.93, 'dmax(mm)': 9.53}

MPIRuns['D2O_Au_CCR05_roomtem'] = {'run': 'run17685', 'temperature': '293K±3K', 'volume(ml)': 0.9, 'height_sample(mm)': 14.3376, 'dmin(mm)': 8.94, 'dmax(mm)': 9.53}
MPIRuns['D2O_Au_CCR05_ice'] = {'run': 'run17686', 'temperature': '243K±3K', 'volume(ml)': 0.9, 'height_sample(mm)': 14.3376, 'dmin(mm)': 8.94, 'dmax(mm)': 9.53}
MPIRuns['H2O_Au_CCR05_roomtem1'] = {'run': 'run17687', 'temperature': '293K±3K', 'volume(ml)': 0.3, 'height_sample(mm)': 4.7792, 'dmin(mm)': 8.94, 'dmax(mm)': 9.53}
MPIRuns['H2O_Au_CCR05_roomtem2'] = {'run': 'run17689', 'temperature': '293K±3K', 'volume(ml)': 0.3, 'height_sample(mm)': 4.7792, 'dmin(mm)': 8.94, 'dmax(mm)': 9.53}
MPIRuns['H2O_Au_CCR05_ice'] = {'run': 'run17690', 'temperature': '243K±3K', 'volume(ml)': 0.3, 'height_sample(mm)': 4.7792, 'dmin(mm)': 8.94, 'dmax(mm)': 9.53}

MPIRuns['H2O_NoAu_CCR05_roomtem'] = {'run': 'run17693', 'temperature': '293K±3K', 'volume(ml)': 0.3, 'height_sample(mm)': 4.7792, 'dmin(mm)': 8.94, 'dmax(mm)': 9.53}
MPIRuns['H2O_NoAu_CCR05_ice1'] = {'run': 'run17694', 'temperature': '243K±3K', 'volume(ml)': 0.3, 'height_sample(mm)': 4.7792, 'dmin(mm)': 8.94, 'dmax(mm)': 9.53}
MPIRuns['H2O_NoAu_CCR05_ice2'] = {'run': 'run17695', 'temperature': '213K±3K', 'volume(ml)': 0.3, 'height_sample(mm)': 4.7792, 'dmin(mm)': 8.94, 'dmax(mm)': 9.53}
MPIRuns['D2O_NoAu_CCR05_roomtem1'] = {'run': 'run17696', 'temperature': '293K±3K', 'volume(ml)': 0.9, 'height_sample(mm)': 14.3376, 'dmin(mm)': 8.94, 'dmax(mm)': 9.53}
MPIRuns['D2O_NoAu_CCR05_roomtem2'] = {'run': 'run17699', 'temperature': '293K±3K', 'volume(ml)': 0.9, 'height_sample(mm)': 14.3376, 'dmin(mm)': 8.94, 'dmax(mm)': 9.53}
MPIRuns['D2O_NoAu_CCR05_ice'] = {'run': 'run17700', 'temperature': '213K±3K', 'volume(ml)': 0.9, 'height_sample(mm)': 14.3376, 'dmin(mm)': 8.94, 'dmax(mm)': 9.53}
MPIRuns['D2O_NoAu_CCR06_roomtem'] = {'run': 'run17701', 'temperature': '302K±3K', 'volume(ml)': 0.9, 'height_sample(mm)': 14.3376, 'dmin(mm)': 8.94, 'dmax(mm)': 9.53}
MPIRuns['D2O_NoAu_CCR06_ice1'] = {'run': 'run17702', 'temperature': '50K±3K', 'volume(ml)': 0.9, 'height_sample(mm)': 14.3376, 'dmin(mm)': 8.94, 'dmax(mm)': 9.53}
MPIRuns['D2O_NoAu_CCR06_ice2'] = {'run': 'run17703', 'temperature': '12.8K±3K', 'volume(ml)': 0.9, 'height_sample(mm)': 14.3376, 'dmin(mm)': 8.94, 'dmax(mm)': 9.53}

ModuleCorner_Exp={}
with open('%s'%findData('MPI2022Exp/moduleCorner_Exp.json')) as jsonFile:
    ModuleCorner_Exp=json.load(jsonFile)  
# print(ModuleCorner_Exp['module10802']['left_bottom']['pos']) 