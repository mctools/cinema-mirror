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

from scipy.spatial import KDTree
points = np.array([[0,0,1],[0,1,0],[1,0,0]])
tree = KDTree(points)
distanace, idx = tree.query([[0, 0, 1.1], [0, 0, 0.1]], k=1)
print(distanace)
print(idx)


class PixelLocator(KDTree):
    def __init__(pixelID, location, tolerence=None):
        super().__init__(location)
        self.pixelID = pixelID
        self.tolerence = tolerence

    def locate(locations, numNearestPt=1):
        dist, idx = self.query(locations, k=numNearestPt)
        #fixme: print warnings or error if dist is
        return self.pixelID[idx], dist
