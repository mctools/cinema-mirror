import pathlib
import random
import sys
import signal
import time
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
    
    mp_id_list = load_completed_mids(data_dir)
    if len(mp_id_list) == 0:
        mp_id_list = find_mp_list_by_dir(data_dir)
    m_completed_list = load_completed_materials(data_dir)
    
    helper = MPHelper(api_key, data_dir=str(data_dir))
    
    for _item in all_elements:
        if __STOPRUN__:
            break
        
        print(f'Process {_item} ...')
        _m = sort_components(_item)
        if _m is None:
            continue
        if _m in m_completed_list:
            print(f'{_m} already processed, skip')
            continue
        
        _r = []
        try:
            time.sleep(random.randint(3, 10))
            _r = helper.query_mids(_m)
        except Exception as ex:
            print(ex)
            continue
        
        _c = 0
        _has_new = False
        for id in _r:
            if __STOPRUN__:
                break
            
            if id in mp_id_list:
                print(f'{id} already downloaded, skip.')
                _c += 1
                continue
            
            try:
                print(f'download: {id} ...')
                helper.download(id)
                mp_id_list.append(id)
                _c += 1
                _has_new = True
                print(f'download success: {id}')
            except Exception as ex:
                print(ex)
            
            if _has_new and _c//10 > 0:
                save_completed_mids(mp_id_list, data_dir)
                    
            time.sleep(random.randint(3, 10))
        
        if _c == len(_r):
            m_completed_list.append(_m)
            save_completed_materials(m_completed_list, data_dir)
             
    if __STOPRUN__:
        save_completed_mids(mp_id_list, data_dir)
        #save_completed_materials(m_completed_list, data_dir)
    
    #r = helper.query( {"elements": {"$in": ["Li", "Na", "K"], "$all": ["O"]}, "nelements": 2}, None)
    #print(r)
__STOPRUN__ = False

def signal_handler(signum, frame):
    global __STOPRUN__
    __STOPRUN__ = True
    print('STOP signal received, will stop soon.')
    
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    run()
