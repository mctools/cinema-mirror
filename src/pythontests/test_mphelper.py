#!/usr/bin/env python

import unittest
import tempfile
import pathlib
from Cinema.PiXiu.io.MPHelper import MPHelper


class MPHelperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = tempfile.TemporaryDirectory()
        cls.helper = MPHelper("__THIS__IS__A__FAKE__KEY", pathlib.Path(cls.data_dir.name))
    
    @classmethod
    def tearDownClass(cls):
        cls.data_dir.cleanup()
        
    def test_api_available(self):
        r = self.helper.query_mids('Cr')
        self.assertTrue(isinstance(r, list), "MP API test fail")
    
if __name__ == '__main__':
    unittest.main()