import pathlib
import json
import requests
import signal
from Cinema.PiXiu.IO.AflowHelper import AflowHelper
from element import ALL_ELEMENTS


# _url ='http://materials.duke.edu/AFLOWDATA/LIB4_RAW/AgAlCoLa:PAW_PBE/ABCD_cF16_216_c_d_b_a.ABCD/?format=json'
# 'http://materials.duke.edu/AFLOWDATA/LIB0_LIB/Ac:PAW_PBE_KIN:06Sep2000/0' # pre-processed
__STOPRUN__ = False

def save_list_to_text(data_list, dir, fname):
    _list = sorted(set(data_list))
    try: 
        with open(dir.joinpath(fname), 'w') as fp:
            for _item in _list:
                fp.write(str(_item) + '\n')
    except Exception as ex:
        print(ex)
        
def load_list_from_text(dir, fname):
    _r = []
    try: 
        with open(dir.joinpath(fname), 'r') as fp:
            for _line in fp:
                _line = _line.strip()
                if len(_line) == 0 or _line.startswith('#'):
                    continue
                _r.append(_line)
    except Exception as ex:
        print(ex)
    
    return _r

def crawl_materials(aflow_helper, element, data_dir):
    global __STOPRUN__
    _summons = f'species({element})'
    _page_offset = 1
    _page_size = 1000
    _completed = True
    
    while True:
        if __STOPRUN__:
            _completed = False
            break
        
        try:
            print(f'Dealing with: {element}, {_page_offset}, {_page_size} ...')
            _r = aflow_helper.query(_summons, _page_offset, _page_size)
            if len(_r) > 0:
                _f = data_dir.joinpath(f'{element}_{_page_offset}_{len(_r)}.json')
                _page_offset += 1
                with open(f'{_f}', 'w') as _fp:
                    json.dump(_r, _fp)
            else:
                break
        except Exception as ex:
            print(ex)
            continue
    
    return _completed
        
            
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
            crawl_properties(_d, format, data_dir.joinpath('temp'), exist_auids)
    
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
            _count += 1
            print(f'{_f} saved')
    
    return exist_auids
            
def run():
    global __STOPRUN__
    
    _home = pathlib.Path.home()
    _data_dir = _home.joinpath('aflow_data')
    _data_dir.joinpath('temp').mkdir(parents=True, exist_ok=True)    
    #base_url = 'http://materials.duke.edu/AFLOWDATA'
    #format = 'format=json'
    _f_auid_completed = 'auids_completed.txt'
    _f_elements_completed = 'elements_completed.txt'
    elements_completed = load_list_from_text(_data_dir, _f_elements_completed)
    auids_completed = load_list_from_text(_data_dir, _f_auid_completed)
    aflow_helper = AflowHelper(_data_dir, timeout=60)
    
    #### Crawl all meterials
    for element in ALL_ELEMENTS:
        if __STOPRUN__:
            break
        
        if element in elements_completed:
            print(f'{element} has been crawled, skip.')
            continue
        
        _r = crawl_materials(aflow_helper, element, _data_dir.joinpath('temp'))
        if _r:
            elements_completed.append(element)
            save_list_to_text(elements_completed, _data_dir, _f_elements_completed)    
           
    save_list_to_text(auids_completed, _data_dir, _f_auid_completed)
            
def signal_handler(signum, frame):
    global __STOPRUN__
    __STOPRUN__ = True
    print('STOP signal received, will stop soon.')
    
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    run()
    print('Done.')
    