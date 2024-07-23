
################################################################################
##                                                                            ##
##  This file is part of Prompt (see https://gitlab.com/xxcai1/Prompt)        ##
##                                                                            ##
##  Copyright 2021-2024 Prompt developers                                     ##
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
from .solid import Box

from .configstr import ConfigString

class Scorer(ConfigString):
    pass

class PSD(ConfigString):
    def __init__(self) -> None:
        super().__init__()
        self.cfg_Scorer='PSD'
        self.cfg_name = 'PSD'
        self.cfg_xmin = -1.
        self.cfg_xmax = 1.
        self.cfg_numbin_x = 10 
        self.cfg_ymin = -1.
        self.cfg_ymax = 1.
        self.cfg_numbin_y = 10 
        self.cfg_ptstate = 'ENTRY' 
        self.cfg_type = 'XZ'

class WlSpectrum(ConfigString):
    def __init__(self) -> None:
        super().__init__()
        self.cfg_Scorer='WlSpectrum'
        self.cfg_name = 'WlSpectrum'
        self.cfg_min = 0.0
        self.cfg_max = 5
        self.cfg_numbin = 100
        self.cfg_ptstate = 'ENTRY'

class ESpectrum(ConfigString):
    def __init__(self) -> None:
        super().__init__()
        self.cfg_Scorer='ESpectrum'
        self.cfg_name = 'ESpectrum'
        self.cfg_scoreTransfer = 0
        self.cfg_min = 1e-5
        self.cfg_max = 0.25
        self.cfg_numbin = 100
        self.cfg_ptstate = 'ENTRY'

class TOF(ConfigString):
    def __init__(self) -> None:
        super().__init__()
        self.cfg_Scorer='TOF'
        self.cfg_name = 'TOF'
        self.cfg_min = 0.0025
        self.cfg_max = 0.008
        self.cfg_numbin = 100
        self.cfg_ptstate = 'ENTRY'

class VolFluence(ConfigString):
    def __init__(self) -> None:
        super().__init__()
        self.cfg_Scorer='VolFluence'
        self.cfg_name = 'VolFluence'
        self.cfg_min = 0
        self.cfg_max = 1
        self.cfg_numbin = 100
        self.cfg_ptstate = 'ENTRY'
        self.cfg_linear = 'yes'

# FIXME: scorer object construction moves from strings to ctype
# the following class name move to *V1
class ScorerHelperV1:
    def __init__(self, name, min, max, numbin, ptstate) -> None:
        self.name = name
        self.min = min
        self.max = max
        self.numbin = numbin
        self.ptstate = ptstate

    def __realinit(self):
        self.score.cfg_name = self.name
        self.score.cfg_min = self.min
        self.score.cfg_max = self.max
        self.score.cfg_numbin = self.numbin
        self.score.cfg_ptstate = self.ptstate
        
    def make(self, vol):
        vol.addScorer(self.score.cfg)

class ESpectrumHelperV1(ScorerHelperV1): 
    def __init__(self, name, min=1e-5, max=1, numbin = 100, ptstate: str = 'ENTRY', energyTransfer=False) -> None:
        super().__init__(name, min, max, numbin, ptstate)
        self.score = ESpectrum()
        if energyTransfer:
            self.score.cfg_scoreTransfer = 1
        else:
            self.score.cfg_scoreTransfer = 0
        self._ScorerHelperV1__realinit()
    
class WlSpectrumHelperV1(ScorerHelperV1): 
    def __init__(self, name, min=0.1, max=10, numbin = 100, ptstate: str = 'ENTRY') -> None:
        super().__init__(name, min, max, numbin, ptstate)
        self.score = WlSpectrum()
        self._ScorerHelperV1__realinit()
    
class TOFHelperV1(ScorerHelperV1): 
    def __init__(self, name, min=0, max=40e-3, numbin = 100, ptstate: str = 'ENTRY') -> None:
        super().__init__(name, min, max, numbin, ptstate)
        self.score = TOF()
        self._ScorerHelperV1__realinit()

class VolFluenceHelperV1(ScorerHelperV1): 
    def __init__(self, name, min=1e-6, max=10, numbin = 100, ptstate: str = 'PEA_PRE', linear = False) -> None:
        super().__init__(name, min, max, numbin, ptstate)
        self.score = VolFluence()
        if linear:
            self.score.cfg_linear = 'yes'
        else: 
            self.score.cfg_linear = 'no'
        self._ScorerHelperV1__realinit()


# DepositionHelper is a special class that skipped the traditional string based initialisation.
# A C++ object is directly created in  
from ..Interface import *
_pt_ScorerDeposition_new = importFunc('pt_ScorerDeposition_new', type_voidp, [type_cstr, type_dbl, type_dbl, type_uint, type_uint, type_int, type_bool, type_int])
_pt_ScorerESpectrum_new = importFunc('pt_ScorerESpectrum_new', type_voidp, [type_cstr, type_bool, type_dbl, type_dbl, type_uint, type_uint, type_int, type_int, type_bool ])
_pt_ScorerTOF_new = importFunc('pt_ScorerTOF_new', type_voidp, [type_cstr, type_dbl, type_dbl, type_uint, type_uint, type_int, type_int ])
_pt_ScorerWlSpectrum_new = importFunc('pt_ScorerWlSpectrum_new', type_voidp, [type_cstr, type_dbl, type_dbl, type_uint, type_uint, type_int, type_int ])
_pt_ScorerVolFluence_new = importFunc('pt_ScorerVolFluence_new', type_voidp, [type_cstr, type_dbl, type_dbl, type_uint, type_dbl, type_uint, type_int, type_bool, type_int])
_pt_ScorerMultiScat_new = importFunc('pt_ScorerMultiScat_new', type_voidp, [type_cstr, type_dbl, type_dbl, type_uint, type_uint, type_int, type_int])
_pt_addMultiScatter = importFunc('pt_addMultiScatter', None, [type_voidp, type_voidp, type_int])

class ScorerHelper:
    def __init__(self, name, min, max, numbin, pdg = 2112, ptstate = 'ENTRY', groupID=0) -> None:
        self.name = name
        self.min = min
        self.max = max
        self.numbin = numbin
        self.pdg = pdg
        self.ptstate = ptstate
        self.groupID = groupID
        self.linear = True

    def __realinit(self):
        self.score.cfg_name = self.name
        self.score.cfg_min = self.min
        self.score.cfg_max = self.max
        self.score.cfg_numbin = self.numbin
        self.score.cfg_ptstate = self.ptstate

    @property
    def ptsNum(self):
        """Enum of particle tracing state.
        """
        if self.ptstate=='SURFACE':
            return 0
        elif self.ptstate=='ENTRY':
            return 1
        elif self.ptstate=='PROPAGATE_PRE':
            return 2
        elif self.ptstate=='PROPAGATE_POST':
            return 3
        elif self.ptstate=='EXIT':
            return 4
        elif self.ptstate=='PEA_PRE':
            return 5
        elif self.ptstate=='PEA_POST':
            return 6
        elif self.ptstate=='ABSORB':
            return 7
        else:
            raise RuntimeError(f'Scorer type {self.ptstate} is undefined.') 
        
    def make(self, vol):
        vol.addScorer(self.score.cfg)


class TOFHelper(ScorerHelper):
    def __init__(self, name : str, min : float = 0., max : float = 40e-3, numbin : int = 100, 
                 pdg : int = 2112, ptstate : str = 'ENTRY', groupID : int = 0) -> None:
        super().__init__(name, min, max, numbin, pdg, ptstate, groupID)

    def make(self, vol):
        cobj = _pt_ScorerTOF_new(self.name.encode('utf-8'),
                                 self.min,
                                 self.max,
                                 self.numbin,
                                 self.pdg,
                                 self.ptsNum,
                                 self.groupID)
        vol.addScorer(self, cobj)

class ESpectrumHelper(ScorerHelper):
    def __init__(self, name, min=1e-5, max=1, numbin = 100, pdg : int = 2112, 
                 ptstate: str = 'ENTRY', energyTransfer=False, groupID=0, linear = False) -> None:
        super().__init__(name, min, max, numbin, pdg, ptstate, groupID)
        self.energyTransfer = energyTransfer
        self.linear = linear

    def make(self, vol):
        cobj = _pt_ScorerESpectrum_new(self.name.encode('utf-8'), 
                                        self.energyTransfer,
                                        self.min,
                                        self.max,
                                        self.numbin,
                                        self.pdg,
                                        self.ptsNum,
                                        self.groupID,
                                        self.linear
                                        )
        vol.addScorer(self, cobj)
        self.cobj = cobj

    def addScatterCounter(self, scatterCounter, scatterNumberRequired):
        _pt_addMultiScatter(scatterCounter.cobj, self.cobj, scatterNumberRequired)

class MultiScatCounter(ScorerHelper):
    def __init__(self) -> None:
        super().__init__(name="ScatterCounter", min=0., max=1., numbin=1, pdg=2112, ptstate='PROPAGATE_POST', groupID=0)

    def make(self, vol):
        cobj = _pt_ScorerMultiScat_new(self.name.encode('utf-8'), 
                                        self.min,
                                        self.max,
                                        self.numbin,
                                        self.pdg,
                                        self.ptsNum,
                                        self.groupID)
        vol.addScorer(self, cobj)
        self.cobj = cobj
        
    
class WlSpectrumHelper(ScorerHelper):
    def __init__(self, name, min=0.1, max=10, numbin = 100, pdg : int = 2112, ptstate : str = 'ENTRY', groupID : int = 0) -> None:
        super().__init__(name, min, max, numbin, pdg, ptstate, groupID)

    def make(self, vol):
        cobj = _pt_ScorerWlSpectrum_new(self.name.encode('utf-8'), 
                                        self.min,
                                        self.max,
                                        self.numbin,
                                        self.pdg,
                                        self.ptsNum,
                                        self.groupID
                                        )
        vol.addScorer(self, cobj)

class VolFluenceHelper(ScorerHelper):
    def __init__(self, name, min = 1e-6, max = 10, numbin = 100, pdg = 2112, linear : bool = False, groupID : int = 0) -> None:
        super().__init__(name, min, max, numbin, pdg, ptstate = 'PEA_PRE', groupID = groupID)
        self.linear = linear

    def make(self, vol):
        volCapacity = vol.getCapacity()
        cobj = _pt_ScorerVolFluence_new(
            self.name.encode('utf-8'), 
            self.min,
            self.max,
            self.numbin,
            volCapacity,
            self.pdg,
            self.ptsNum,
            self.linear,
            self.groupID
        )
        vol.addScorer(self, cobj)

class DepositionHelper(ScorerHelper):
    def __init__(self, name : str, min, max, numbin, pdg, ptstate, linear : bool = False, groupID : int = 0) -> None:
        super().__init__(name, min, max, numbin, pdg, ptstate, groupID)
        self.linear = linear

    def make(self, vol):
        # enum class ScorerType {SURFACE, ENTRY, PROPAGATE, EXIT, PEA, ABSORB};
        cobj = _pt_ScorerDeposition_new(self.name.encode('utf-8'), 
                                        self.min,
                                        self.max,
                                        self.numbin,
                                        self.pdg,
                                        self.ptsNum,
                                        self.linear,
                                        self.groupID)
        vol.addScorer(self, cobj)



# fixme: what is the function followed used for??? x.
def makePSD(name, vol, numbin_dim1=1, numbin_dim2=1, ptstate : str = 'ENTRY', type : str = 'XY'):
    if not isinstance(vol.solid, Box):
        raise TypeError('makePSD only used for "Box" type volume')
    det = PSD()
    det.cfg_name = name
    det.cfg_numbin_x = numbin_dim1 
    det.cfg_numbin_y = numbin_dim2 
    det.cfg_ptstate = ptstate
    det.cfg_type = type
    if type == 'XY':
        det.cfg_xmin = -vol.solid.hx
        det.cfg_xmax = vol.solid.hx
        det.cfg_ymin = -vol.solid.hy
        det.cfg_ymax = vol.solid.hy
    elif type == 'XZ':
        det.cfg_xmin = -vol.solid.hx
        det.cfg_xmax = vol.solid.hx
        det.cfg_ymin = -vol.solid.hz
        det.cfg_ymax = vol.solid.hz
    elif type == 'YZ':
        det.cfg_xmin = -vol.solid.hy
        det.cfg_xmax = vol.solid.hy
        det.cfg_ymin = -vol.solid.hz
        det.cfg_ymax = vol.solid.hz
    vol.addScorer(det.cfg)

        
