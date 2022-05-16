import pathlib
import random
import sys
import signal
import time
import json
from Cinema.PiXiu.IO.MPHelper import MPHelper


def find_mp_list_by_dir(dir):
    _r = []
    _fs = sorted(dir.glob('*.json'))
    for _item in _fs:
        _r.append(_item.stem.split('_')[0])
    
    return _r

def load_completed_materials(dir):
    _r = []
    try: 
        with open(dir.joinpath('m_completed.txt'), 'r') as fp:
            for _line in fp:
                _line = _line.strip()
                if len(_line) == 0 or _line.startswith('#'):
                    continue
                _r.append(_line)
    except Exception as ex:
        print(ex)
    
    return _r

def load_completed_mids(dir):
    _r = []
    try: 
        with open(dir.joinpath('mids_completed.txt'), 'r') as fp:
            for _line in fp:
                _line = _line.strip()
                if len(_line) == 0 or _line.startswith('#'):
                    continue
                _r.append(_line)
    except Exception as ex:
        print(ex)
    
    return _r

def load_todo_mids(dir):
    _r = []
    try: 
        with open(dir.joinpath('mids_todo.txt'), 'r') as fp:
            for _line in fp:
                _line = _line.strip()
                if len(_line) == 0 or _line.startswith('#'):
                    continue
                _r.append(_line)
    except Exception as ex:
        print(ex)
    
    return sorted(set(_r))

def save_todo_mids(mids_list, dir):
    _list = sorted(set(mids_list))
    try: 
        with open(dir.joinpath('mids_todo.txt'), 'w') as fp:
            for _item in _list:
                fp.write(str(_item) + '\n')
    except Exception as ex:
        print(ex)

def save_completed_mids(mids_list, dir):
    _list = sorted(set(mids_list))
    try: 
        with open(dir.joinpath('mids_completed.txt'), 'w') as fp:
            for _item in _list:
                fp.write(str(_item) + '\n')
    except Exception as ex:
        print(ex)
        
def save_completed_materials(m_list, dir):
    _list = sorted(set(m_list))
    try: 
        with open(dir.joinpath('m_completed.txt'), 'w') as fp:
            for _item in _list:
                fp.write(str(_item) + '\n')
    except Exception as ex:
        print(ex)
        
def sort_components(compstr):
    if compstr is None or len(compstr.strip()) == 0:
        return None

    _t = sorted(set(compstr.strip('-').split('-')))
    return '-'.join(_t)

def scrab_mp_structures(mphelper, data_dir, page_offset=0, page_limit=500):
    for _f in data_dir.glob('*.json'):
        _t = _f.stem.split('_')
        if len(_t) >= 3:
            page_offset = max(page_offset, int(_t[1]))
            page_limit = max(page_limit, int(_t[2]))
        elif len(_t) == 2:
            page_offset = max(page_offset, int(_t[1]))
    page_offset = page_offset + page_limit
    print(f'start with: page_offset={page_offset}, page_limit={page_limit}')
    mphelper.get_structures_by_optimade(page_offset, page_limit)
    
def parse_optimade_json(data_dir):
    _mpids = []
    for _f in data_dir.glob('*.json'):
        print(f'read: {_f}')
        with open(_f, 'r') as _fp:
            _m = json.load(_fp)
            for _item in _m['data']:
                _mpids.append(_item['id'])
    
    _mpids = sorted(set(_mpids))
    print(f'total {len(_mpids)} found')
    
    return _mpids

def query_mpids_by_components(mphelper, component_list, m_completed_list=[]):
    global __STOPRUN__
    _mids = []
    
    for _item in component_list:
        if __STOPRUN__:
            break
        
        print(f'Process {_item} ...')
        _m = sort_components(_item)
        if _m is None:
            continue
        if _m in m_completed_list:
            print(f'{_m} already processed, skip')
            continue
        
        
        try:
            time.sleep(random.randint(3, 10))
            _mids += mphelper.query_mids(_m)
            m_completed_list.append(_m)
        except Exception as ex:
            print(ex)
    
    return _mids, m_completed_list

def reduce_list(origin_list, reduce_elems):
    for _v in reduce_elems:
        try:
            origin_list.remove(_v)
        except ValueError:
            pass
        
    return origin_list

all_elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al']
all_elements += ['Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn']
all_elements += ['Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr']
all_elements += ['Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag']
all_elements += ['Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba']
all_elements += ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
all_elements += ['Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po']
all_elements += ['At', 'Rn', 'Fr', 'Ra']
all_elements += ['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
all_elements += ['Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn']

def run():
    global __STOPRUN__
    #api_key='SOME_KEY'        
    api_key = None
    data_dir = pathlib.Path.home().joinpath('mpdata')
    
    mp_id_completed_list = load_completed_mids(data_dir)
    if len(mp_id_completed_list) == 0:
        mp_id_completed_list = find_mp_list_by_dir(data_dir)
    m_completed_list = load_completed_materials(data_dir)
    mp_id_todo_list = load_todo_mids(data_dir)
    
    helper = MPHelper(api_key, data_dir=str(data_dir), timeout=5)
    
    # Scrab structures through optimade. All data before 2021-3-18.
    #opd_dir = data_dir.joinpath('optimade')
    #scrab_mp_structures(helper, opd_dir)
    #mp_id_todo_list += parse_optimade_json(opd_dir)
 
    # Scrab mpids by components       
    #mpids, m_completed_list = query_mpids_by_components(helper, all_elements, m_completed_list)        
    #save_completed_materials(m_completed_list, data_dir)
    #mp_id_todo_list += mpids
    #save_todo_mids(mp_id_todo_list, data_dir)    
        
    mp_id_todo_list = reduce_list(mp_id_todo_list, mp_id_completed_list)
    print(f'Total {len(mp_id_todo_list)} to be download.')
    _c = 1
    for id in mp_id_todo_list:
        if __STOPRUN__:
            break
        
        try:
            print(f'download: {id} ...')
            helper.download(id)
            mp_id_completed_list.append(id)
            _c += 1
            print(f'download success: {id}')
        except Exception as ex:
            print(ex)
        
        if _c % 20 == 0:
            save_completed_mids(mp_id_completed_list, data_dir)
                
        time.sleep(random.randint(3, 10))
             

    mp_id_todo_list = reduce_list(mp_id_todo_list, mp_id_completed_list)
    save_completed_mids(mp_id_completed_list, data_dir)
    save_todo_mids(mp_id_todo_list, data_dir)
    print(f'Total {_c - 1} materials downloaded.')


__STOPRUN__ = False

def signal_handler(signum, frame):
    global __STOPRUN__
    __STOPRUN__ = True
    print('STOP signal received, will stop soon.')
    
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    run()
    print('Done.')
