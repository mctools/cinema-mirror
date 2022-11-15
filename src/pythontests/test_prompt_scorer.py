#!/usr/bin/env python3

import numpy as np
import os
from Cinema.Prompt import PromptFileReader

<<<<<<< HEAD
scorertest_dict ={}
scorertest_dict['ESpectrum'] = {'gdml': 'ESpectrum.gdml', 'mcpl': 'ScorerESpectrum_detector_seed113.mcpl.gz', 'value': [1.27765, 89., 89. ]}
scorertest_dict['MultiScat'] = {'gdml': 'MultiScat.gdml', 'mcpl': 'ScorerMultiScat_D2O_seed113.mcpl.gz', 'value': [18., 392., 392.]}
scorertest_dict['NeutronSq'] = {'gdml': 'NeutronSq.gdml', 'mcpl': 'ScorerNeutronSq_SofQ_seed113.mcpl.gz', 'value': [25275.25, 174., 174.]}
scorertest_dict['RotatingObj'] = {'gdml': 'RotatingObj.gdml', 'mcpl': 'ScorerRotatingObj_roen_seed113.mcpl.gz', 'value': [50.49999999999999, 4156., 4156.]}
scorertest_dict['TOF'] = {'gdml': 'TOF.gdml', 'mcpl': 'ScorerTOF_detector_seed113.mcpl.gz', 'value': [250.25, 797., 797.]}
scorertest_dict['VolFlux'] = {'gdml': 'VolFlux.gdml', 'mcpl': 'ScorerVolFlux_Sflux_seed113.mcpl.gz', 'value': [6.060050500000000e+01, 5.081422907074905e-03, 1.455600000000000e+04]}
scorertest_dict['PSD'] = {'gdml': 'PSD.gdml', 'mcpl': 'ScorerPSD_NeutronHistMap_seed113.mcpl.gz', 'value': [0., 0., 818., 818.]}
scorertest_dict['guide'] = {'gdml': 'guide.gdml', 'mcpl': 'ScorerPSD_Monitor2_seed113.mcpl.gz', 'value': [0., 0., 207.18374089553834, 235.]}
=======
gdml_filelist=['ESpectrum.gdml', 'MultiScat.gdml', 'NeutronSq.gdml', 'RotatingObj.gdml', 'TOF.gdml', 'VolFlux.gdml', 'PSD.gdml', 'guide.gdml']
mcpl_filelist=['ScorerESpectrum_detector_seed113.mcpl.gz', 'ScorerMultiScat_D2O_seed113.mcpl.gz', 'ScorerNeutronSq_SofQ_seed113.mcpl.gz', 'ScorerRotatingObj_roen_seed113.mcpl.gz', 'ScorerTOF_detector_seed113.mcpl.gz', 'ScorerVolFlux_Sflux_seed113.mcpl.gz','ScorerPSD_NeutronHistMap_seed113.mcpl.gz','ScorerPSD_Monitor2_seed113.mcpl.gz']
valuelist=[[1.27765, 89., 89. ], [18., 392., 392.], [25275.25, 174., 174.], [50.49999999999999, 4156., 4156.], [250.25, 797., 797.], [6.060050500000000e+01, 5.081422907074905e-03, 1.455600000000000e+04], [0., 0., 818., 818.],[0., 0., 207.18374089553834, 235.]]
>>>>>>> f1e30b97a4d38cf46103806ab84be9017ae124aa

for scorername in scorertest_dict:
    scorer = scorertest_dict[scorername]
    os.system('rm %s'%scorer['mcpl'])
    os.system('prompt -g %s -s 113 -n 1e3'%scorer['gdml'])
    
    f = PromptFileReader(scorer['mcpl'])
    hist_weight = f.getData('content').sum()
    hist_hit = f.getData('hit').sum()
    if(scorername=='PSD' or scorername=='guide'):
        hist_xedge = f.getData('xedge').sum()
        hist_yedge = f.getData('yedge').sum()
        res = np.array([hist_xedge, hist_yedge, hist_weight, hist_hit])
    else:
        hist_edge = f.getData('edge').sum()
        res = np.array([hist_edge, hist_weight, hist_hit])
    
    np.set_printoptions(precision=16)
    print(res)
    np.testing.assert_allclose(res, scorer['value'], rtol=1e-15)

print('passed prompt_scorer test')
