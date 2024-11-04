#!/usr/bin/env python3
import os
import json

class JsonParser():
    def __init__(self, fname='section.json'):
        self.fname = fname
        self.ginfo = self.readIn()

    def readIn(self, ifprint=False):
        fname = os.path.join(os.getcwd(), self.fname)
        with open(fname, 'r') as f:
            eell = json.load(f)
        if ifprint:
            print(eell)
        return eell.values()

    def get_bounds(self, var_list):
        zinit = False
        for zz in var_list:
            if zinit == False:
                zmin = zmax = zz
                zinit = True
                continue
            if zz > zmax:
                zmax = zz
            if zz < zmin:
                zmin = zz
        zmidpoint = 0.5 * (zmin + zmax)
        return zmin, zmax, zmidpoint
        
    def bound_zinfo(self):
        var_list = []
        for var in self.ginfo:
            var_list.append(var['zlocation'] + var['length'])
        return self.get_bounds(var_list)

    def bound_xinfo(self):
        var_list = []
        for var in self.ginfo:
            var_list.append(var['entryOpening']['halfwidth'])
        return self.get_bounds(var_list)

    def bound_yinfo(self):
        var_list = []
        for var in self.ginfo:
            var_list.append(var['entryOpening']['halfheight'])
        return self.get_bounds(var_list)