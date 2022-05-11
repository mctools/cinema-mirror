import os
import json
from pathlib import Path
import requests
import random
from zipfile import ZipFile
from io import BytesIO
from enum import Enum, unique


@unique
class OutputType(Enum):
    POSCAR = 'POSCAR'
    CIF = 'CIF'
    VASP = 'VASP'
    CSSR = 'CSSR'
    JSON = 'JSON'

class MPHelper:
    def __init__(self, api_key=None, data_dir=None, timeout=5):
        self.download_url = 'https://materialsproject.org/materials/download'
        self.mids_url = 'https://materialsproject.org/rest/v2/materials/$QUERYSTR$/mids'
        self.query_url = 'https://materialsproject.org/rest/v2/query'
        self.api_key = api_key
        self.timeout = timeout
        self.data_dir = data_dir
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
        if self.data_dir is not None:
            #os.makedirs(self.data_dir, exist_ok=True)
            Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        if self.api_key is None:
            self.api_key = os.environ.get('MP_API_KEY', None)
        if self.api_key is None:
            raise Exception('MP_API_KEY not set')
    
    def query_mids(self, material):
        if material is None or len(material.strip()) == 0:
            raise Exception('material is required.') 
        url = self.mids_url.replace('$QUERYSTR$', material.strip())
        headers = {
            'User-Agent': (self.user_agent)[random.randint(0, len(self.user_agent) - 1)]
        }
        response = requests.request("GET", url, headers=headers, timeout=self.timeout)
        if response.status_code != requests.codes.ok:
            raise IOError(f'Query mids failed: {response.status_code}')
        
        return (response.json()['response'])
    
    def query(self, criteria, properties=None):
        valid_properties = ['elements', 'nelements', 'nsites', 'formula', 'normalized_formula',
                            'energy', 'energy_per_atom', 'density', 'e_above_hull', 
                            'formation_energy_per_atom', 'material_id']
        
        if not isinstance(criteria, dict):
            raise IOError('criteria is not a dict')
        if len(criteria) == 0:
            raise IOError('criteria must not empty')
        for key in criteria.keys():
            if key not in valid_properties:
                raise IOError(f'unknown property in criteria: {key}')
            
        if properties is None:
            properties = valid_properties
        elif isinstance(properties, str):
            properties = [properties]
        elif isinstance(properties, list):
            properties = properties
        else:
            raise IOError('properties must be str or list')
        
        for key in properties:
            if key not in valid_properties:
                raise IOError(f'unknown property in properties: {key}')
        
        headers = {
            'X-API-KEY': self.api_key,
            'User-Agent': (self.user_agent)[random.randint(0, len(self.user_agent) - 1)]
        }
        payload = {'criteria': json.dumps(criteria), 'properties': json.dumps(properties)}
        response = requests.request("POST", self.query_url, headers=headers, data=payload, timeout=self.timeout)
        if response.status_code != requests.codes.ok:
            raise IOError(f'Query failed: {response.status_code}')
        
        return(response.json()['response'])
        
    def download(self, material_id, output='JSON', cif_type = 'conventional_standard'):
        if material_id is None:
            raise IOError('material id not provided.')
        
        material_id = str(material_id).lower()
        if material_id.find('-') < 0:
            material_id = 'mp-' + material_id    
                    
        if isinstance(output, OutputType):
            output = output.value
            
        payload = f'material_id={material_id}&output={output}&cifType={cif_type}'
        headers = {
            #'X-API-KEY': self.api_key,
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept-Encoding': 'gzip, deflate, br',
            'User-Agent': (self.user_agent)[random.randint(0, len(self.user_agent) - 1)]
        }

        response = requests.request("POST", self.download_url, headers=headers, data=payload, timeout=self.timeout)
        if response.status_code != requests.codes.ok:
            raise IOError(f'Download failed: {response.status_code}')

        headers = response.headers
        if headers['Content-Type'] != 'application/zip' or headers['Content-Encoding'] != 'gzip':
            raise IOError(f'Download failed: response is not a gzip file.')

        #file_name = headers['Content-Disposition'].replace('filename=', '')                
        
        #with BytesIO() as buffer:
            #for chunk in response.iter_content(chunk_size=1024, decode_unicode=False):
            #    buffer.write(chunk)
        with BytesIO(response.content) as buffer:
            with ZipFile(buffer) as z:
                z.extractall(path=self.data_dir)
