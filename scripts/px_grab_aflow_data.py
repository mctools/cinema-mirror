import pathlib
import json
import requests
import signal
from Cinema.PiXiu.IO.AflowHelper import AflowHelper
from element import ALL_ELEMENTS


# 'http://materials.duke.edu/AFLOWDATA/LIB4_RAW/AgAlCoLa:PAW_PBE/ABCD_cF16_216_c_d_b_a.ABCD/?format=json'
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

def save_dict_to_text(data_dict, dir, fname):
    try: 
        with open(dir.joinpath(fname), 'w') as fp:
            for _k, _v in data_dict.items():
                fp.write(str(_k) + ',' + str(_v) + '\n')
    except Exception as ex:
        print(ex)
        
def load_dict_from_text(dir, fname):
    _r = {}
    try: 
        with open(dir.joinpath(fname), 'r') as fp:
            for _line in fp:
                _line = _line.strip()
                if len(_line) == 0 or _line.startswith('#'):
                    continue
                
                _t = _line.split(r',')
                _r[(_t[0]).strip()] = (_t[1]).strip()
    except Exception as ex:
        print(ex)
    
    return _r

def reduce_dict(origin_dict, reduce_obj):
    _keys = []
    if isinstance(reduce_obj, dict):
        _keys = reduce_obj.keys()
    elif isinstance(reduce_obj, list):
        _keys = reduce_obj
    else:
        return origin_dict
    
    for _k in _keys:
        try:
            del origin_dict[_k]
        except Exception as ex:
            print(ex)
    
    return origin_dict

def gen_initial_todo_dict(dir, pattern='*.json'):
    _r = {}
    _fs = sorted(dir.glob(pattern))
    for _f in _fs:
        print(f'Reading {_f} ...')
        with open(_f, 'r') as _fp:
            _docs = json.load(_fp)
            for _doc in _docs:
                _r[_doc['auid'].replace(r'aflow:', r'', 1)] = _doc['aurl']
                    
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
            
def run():
    global __STOPRUN__
    
    _home = pathlib.Path.home()
    _data_dir = _home.joinpath('aflow_data')
    _data_dir.joinpath('temp').mkdir(parents=True, exist_ok=True)    
    #base_url = 'http://materials.duke.edu/AFLOWDATA'
    #format = 'format=json'
    _f_auid_completed = 'auids_completed.txt'
    _f_elements_completed = 'elements_completed.txt'
    _f_auids_todo = 'auids_todo.txt'
    _todo_dict = {}
     
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
    
    #### Prepare auids to be download.
    if _data_dir.joinpath(_f_auids_todo).exists():
        print(f'Loading todo auids from file ...')
        _todo_dict = load_dict_from_text(_data_dir, _f_auids_todo)
    else:
        print(f'File not found: {_f_auids_todo}. Begin scan ...')
        _todo_dict = gen_initial_todo_dict(_data_dir.joinpath('temp'))
        print(f'Total {len(_todo_dict)} auids found.')
        print(f'Dereplicate completed auids ...')
        _todo_dict = reduce_dict(_todo_dict, auids_completed)
        print(f'Save todo auids to file ...')
        save_dict_to_text(_todo_dict, _data_dir, _f_auids_todo)
    print(f'Total auids to do: {len(_todo_dict)}.')
    
    #### Download properties
    _c = 1 
    for _auid, _aurl in _todo_dict.items():
        if __STOPRUN__:
            break
        
        try:
            print(f'Downloading {_auid}: {_aurl} ...')
            aflow_helper.download(_aurl, _auid)
            _c += 1
            auids_completed.append(_auid)
        except Exception as ex:
            print(ex)
            
        if _c % 21 == 0:
            save_list_to_text(auids_completed, _data_dir, _f_auid_completed)
           
    print('Save completed auids to file ...')
    save_list_to_text(auids_completed, _data_dir, _f_auid_completed)
    _todo_dict = reduce_dict(_todo_dict, auids_completed)
    print('Save todo auids to file ...')
    save_dict_to_text(_todo_dict, _data_dir, _f_auids_todo)
            
def signal_handler(signum, frame):
    global __STOPRUN__
    __STOPRUN__ = True
    print('STOP signal received, will stop soon.')
    
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    run()
    print('Done.')
    