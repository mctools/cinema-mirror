#!/usr/bin/env python3

import numpy as np
import os
from Cinema.Prompt import PromptFileReader

gdml_filelist=['ESpectrum.gdml', 'MultiScat.gdml', 'NeutronSq.gdml', 'RotatingObj.gdml', 'TOF.gdml', 'VolFlux.gdml', 'PSD.gdml', 'guide.gdml']
mcpl_filelist=['ScorerESpectrum_detector_seed113.mcpl.gz', 'ScorerMultiScat_D2O_seed113.mcpl.gz', 'ScorerNeutronSq_SofQ_seed113.mcpl.gz', 'ScorerRotatingObj_roen_seed113.mcpl.gz', 'ScorerTOF_detector_seed113.mcpl.gz', 'ScorerVolFlux_Sflux_seed113.mcpl.gz','ScorerPSD_NeutronHistMap_seed113.mcpl.gz','ScorerPSD_Monitor2_seed113.mcpl.gz']
valuelist=[[1.27765, 89., 89. ], [18., 392., 392.], [25275.25, 174., 174.], [50.49999999999999, 5480., 5480.], [250.25, 797., 797.], [6.060050500000000e+01, 5.081422907074905e-03, 1.455600000000000e+04], [0., 0., 818., 818.],[0., 0., 207.18374089553834, 235.]]

for i in range(0,8):
    f1=mcpl_filelist[i]
    os.system(f'rm {f1}')
    os.system('prompt -g %s -s 113 -n 1e3'%gdml_filelist[i])

    f = PromptFileReader(f1)
    hist_weight = f.getData('content').sum()
    hist_hit = f.getData('hit').sum()
    if(i>5):
        hist_xedge = f.getData('xedge').sum()
        hist_yedge = f.getData('yedge').sum()
        res = np.array([hist_xedge, hist_yedge, hist_weight, hist_hit])
        
    else:
        hist_edge = f.getData('edge').sum()
        res = np.array([hist_edge, hist_weight, hist_hit])
           
    np.set_printoptions(precision=16)
    print(res)
    np.testing.assert_allclose(res, valuelist[i], rtol=1e-15)

print('passed prompt_scorer test')

