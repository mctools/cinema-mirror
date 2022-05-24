import pathlib
import json
import requests
import signal


#_url ='http://materials.duke.edu/AFLOWDATA/LIB4_RAW/AgAlCoLa:PAW_PBE/ABCD_cF16_216_c_d_b_a.ABCD/?format=json'
__STOPRUN__ = False

def save_completed_auids(auids_list, dir):
    _list = sorted(set(auids_list))
    try: 
        with open(dir.joinpath('auids_completed.txt'), 'w') as fp:
            for _item in _list:
                fp.write(str(_item) + '\n')
    except Exception as ex:
        print(ex)
        
def load_completed_auids(dir):
    _r = []
    try: 
        with open(dir.joinpath('auids_completed.txt'), 'r') as fp:
            for _line in fp:
                _line = _line.strip()
                if len(_line) == 0 or _line.startswith('#'):
                    continue
                _r.append(_line)
    except Exception as ex:
        print(ex)
    
    return _r

def crawl_components(base_url, format):
    for i in range(10):
        _url = f'{base_url}/LIB{i}_RAW/?{format}'
        print(f'Get {_url} ...')
        response = requests.request('GET', _url, timeout=10)
        if response.status_code != requests.codes.ok:
            raise IOError(f'Request failed: {response.status_code}')
        _r = response.json()
        _f = _data_dir.joinpath('temp').joinpath(f'aflow_data_lib_{i}.json')
        with open(f'{_f}', 'w') as _fp:
            json.dump(_r, _fp)
            
def crawl_properties(doc, format, data_dir, exist_auids = []):
    global __STOPRUN__
    _parent_url = doc.get('aurl')
    _parent_url = _parent_url.replace(r'aflowlib.duke.edu:', r'http://materials.duke.edu')
    if doc.get('aflowlib_entries_number') is not None:
        _rets = doc.get('aflowlib_entries')
        for _item in _rets:
            if __STOPRUN__:
                break
            _url = _parent_url + f'/{_item}'
            print(f'Request {_url}/?{format} ...')
            response = requests.request('GET', f'{_url}/?{format}', timeout=10)
            if response.status_code != requests.codes.ok:
                print(f'Request failed: {response.status_code}')
                continue
            _d = response.json()
            crawl_properties(_d, format, data_dir, exist_auids)
    
    if doc.get('aflowlib_entries_number') is not None:   # fix manual Crtl + C
        return exist_auids
                
    _auid = doc.get('auid').replace(r'aflow:', '')
    if _auid in exist_auids:
        print(f'{_auid} already exists. Skip.')
        return exist_auids
    
    _t = _parent_url[_parent_url.rfind('/')+1:]
    _fn = f'{_t}_{_auid}.json'
    _f = data_dir.joinpath(_fn)
    with open(f'{_f}', 'w') as _fp:
            json.dump(doc, _fp)
            exist_auids.append(_auid)
            print(f'{_f} saved')
    return exist_auids
            
def run():
    global __STOPRUN__
    
    _home = pathlib.Path.home()
    _data_dir = _home.joinpath('aflow_data')
    _data_dir.joinpath('temp').mkdir(parents=True, exist_ok=True)    
    base_url = 'http://materials.duke.edu/AFLOWDATA'
    format = 'format=json'
    auids_completed = load_completed_auids(_data_dir)
    
    #crawl_components(base_url, format)
    
    _files = sorted(_data_dir.joinpath('temp').glob('*.json'))
    for _i in _files:
        if __STOPRUN__:
            break
                
        with open(_i, 'r') as _fp:
            _doc = json.load(_fp)
            auids_completed = crawl_properties(_doc, format, _data_dir, auids_completed)
            save_completed_auids(auids_completed, _data_dir)
            
    save_completed_auids(auids_completed, _data_dir)
            
def signal_handler(signum, frame):
    global __STOPRUN__
    __STOPRUN__ = True
    print('STOP signal received, will stop soon.')
    
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    run()
    print('Done.')
    