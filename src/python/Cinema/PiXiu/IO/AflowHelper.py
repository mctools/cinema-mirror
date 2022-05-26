import os
import json
import pathlib
import requests
import random
from enum import Enum, unique


@unique
class OutputType(Enum):
    NONE = 'none'
    DUMP = 'dump'
    PHP = 'pnp'
    HTML = 'html'
    JSON = 'json'
    TEXT = 'text'

class AflowHelper:
    def __init__(self, data_dir, timeout=30):
        self.api_url = 'https://aflow.org/API/aflux'
        self.timeout = timeout
        if isinstance(data_dir, str):
            self.data_dir = pathlib.Path(data_dir)
        else:
            self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.user_agent = ['Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36',
                           'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36',
                           'Mozilla/5.0 (X11; CrOS x86_64 10066.0.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36',
                           'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:70.0) Gecko/20100101 Firefox/70.0',
                           'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:70.0) Gecko/20100101 Firefox/70.0',
                           'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:99.0) Gecko/20100101 Firefox/99.0',
                           'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.97 Safari/537.36 OPR/65.0.3467.48',
                           'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.97 Safari/537.36 OPR/65.0.3467.48',
                           'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Safari/605.1.15',
                           'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36 Edg/101.0.100.0',
                           'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/604.1 Edg/101.0.100.0']               
    
    def query_available_properties(self):
        _url = f'{self.api_url}/?help(properties),format(json)'
        headers = {
            'User-Agent': (self.user_agent)[random.randint(0, len(self.user_agent) - 1)]
        }
        response = requests.request("GET", _url, headers=headers, timeout=self.timeout)
        
        if response.status_code != requests.codes.ok:
            raise IOError(f'Query mids failed: {response.status_code}')
        
        _type = (response.headers)['Content-Type']
        if r'application/json' not in _type:
            raise IOError(f'Not a json response: {_type}')
        
        return response.json()
    
    def query(self, summons, page_offset=1, page_size=64):
        if page_offset == 0:
            _paging = '$paging(0)'
        else:
            _paging = f'$paging({page_offset}, {page_size})'
            
        _url = f'{self.api_url}/?{summons},{_paging},format(json)'
        headers = {
            'User-Agent': (self.user_agent)[random.randint(0, len(self.user_agent) - 1)]
        }
        response = requests.request("GET", _url, headers=headers, timeout=self.timeout)
        if response.status_code != requests.codes.ok:
            raise IOError(f'Query failed: {response.status_code}')
        
        return response.json()
        
    def download(self, aurl, auid, fname=None):
        if aurl is None or len(aurl.strip()) == 0:
            raise IOError('aurl not provided.')        
        
        _t = aurl.replace(r':', r'/', 1)
        _url = f'https://{_t}/?format=json'
        headers = {
            'User-Agent': (self.user_agent)[random.randint(0, len(self.user_agent) - 1)]
        }
        response = requests.request("GET", _url, headers=headers, timeout=self.timeout)
        if response.status_code != requests.codes.ok:
            raise IOError(f'Download failed: {response.status_code}')

        headers = response.headers
        if r'application/json' not in headers['Content-Type']:
            raise IOError(f'Download failed: not a json response.')        
        
        _fn = fname
        if _fn is None:
            _fn = aurl[aurl.rindex(r'/') + 1:]
            auid = auid.replace(r'aflow:', r'', 1)
            _fn = f'{_fn}_{auid}.json'
        with open(self.data_dir.joinpath(_fn), 'w') as _fp:
            json.dump(response.json(), _fp)
                
