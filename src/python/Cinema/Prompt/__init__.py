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


from . import Launcher
from .Launcher import *

from . import PromptFileReader
from .PromptFileReader import *

from . import Mesh
from .Mesh import *

from . import Visualiser
from .Visualiser import *

from . import Histogram
from .Histogram import *

from .geo import *


# __all__ = Launcher.__all__
# __all__ += PromptFileReader.__all__
# __all__ += Mesh.__all__
# __all__ += Visualiser.__all__
__all__ = Histogram.__all__


# from . import Launcher, Visualiser
# from .geo import Box, Volume, Transformation3D, Tessellated
from .gun import PythonGun
from mpi4py import MPI
import numpy as np

class Prompt:
    def __init__(self, seed : int = 4096) -> None:
        self.l = Launcher()
        self.scorer = {}
        self.l.setSeed(seed)
        self.l.setWorld(self.makeWorld())

    def makeWorld(self):
        raise NotImplementedError('') 

    def setGun(self, gun):
        if isinstance(gun, str):
            self.l.setGun(gun)
        else:
            self.l.setPythonGun(gun)
     
    def show(self, num : int = 0):
        self.l.showWorld(num)

    def simulate(self, num : int = 0, timer=True, save2Disk=False):
        self.l.go(int(num), timer=timer, save2Dis=save2Disk)

    def getScorerHist(self, cfg):
        return self.l.getHist(cfg)
    


class PromptMPI(Prompt):
    def __init__(self, seed=4096) -> None:
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        super().__init__(seed+self.rank)
        

    def simulate(self, num : int = 0):
        batchSize = int(num/self.size)
        if self.rank:
          super().simulate(batchSize,  timer=False)
        else:
          super().simulate(num-batchSize*(self.size-1))

    def show(self, num : int = 0):
        if self.rank==0:
            self.l.showWorld(num)
            self.comm.Barrier()
        else:
            self.comm.Barrier()

    def getScorerHist(self, cfg, dst=0):
        hist = super().getScorerHist(cfg)
        weight = hist.getWeight()
        hit = hist.getHit()
        print(f'rank {self.rank} hist info: {hist.getWeight().sum()} {hist.getHit().sum()}')

        recvw = None
        recvh = None

        if self.rank == dst: # only create the buffer for rank0
            recvw = np.empty(weight.size, dtype='float')
            recvh = np.empty(hit.size, dtype='float')
        self.comm.Reduce(weight, recvw, op = MPI.SUM, root=dst)
        self.comm.Reduce(hit, recvh, op = MPI.SUM, root=dst)

        if self.rank == dst:
            hist.setHit(recvh)
            hist.setWeight(recvw)
        return hist
