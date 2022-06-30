#!/usr/bin/env python

import unittest
import tempfile
import pathlib
from Cinema.PiXiu.io.AflowHelper import AflowHelper


class AflowHelperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = tempfile.TemporaryDirectory()
        cls.helper = AflowHelper(pathlib.Path(cls.data_dir.name))
    
    @classmethod
    def tearDownClass(cls):
        cls.data_dir.cleanup()
        
    def test_api_available(self):
        r = self.helper.query_available_properties()
        self.assertTrue("ael_applied_pressure" in r.keys(), "Aflow API test fail")
    
if __name__ == '__main__':
    unittest.main()