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

#
# import time
#
from .Launcher import Launcher as Launcher
from .Mesh import Mesh as Mesh
from .Visualiser import Visualiser as Visualiser

import mcpl
from io import BytesIO
import numpy as np

class PromptFileReader:
    def __init__(self, fn, particleBlocklength=10000, dumpHeader=True):
        self.pfile = mcpl.MCPLFile(fn)
        self.particleBlocklength = particleBlocklength
        if dumpHeader:
            self.pfile.dump_hdr()
            print("comments:\n", self.getComments())

    def dataKeys(self):
        return self.pfile.blobs.keys()

    def getData(self, k):
        raw=BytesIO(self.pfile.blobs[k])
        return np.load(raw)

    def getComments(self):
        return self.pfile.comments

    # this can be used like:
    # for p in reader.blockIterator():
    #     print( p.x, p.y, p.z, p.ekin )
    def blockIterator(self):
     return self.pfile.particle_blocks

    def particleIterator(self):
     return self.pfile.particle_blocks

