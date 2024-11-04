#!/usr/bin/env python3

import h5py, os
import numpy as np
import matplotlib.pyplot as plt



mcdata_dir = './mcstas'
mcdata_fn = 'mccode.h5'

mc_h5file = h5py.File(f'{mcdata_dir}/{mcdata_fn}')
mcstas_data = mc_h5file['entry1']['data']

print(list(mcstas_data.keys()))




# plot entry exit energy
for entry_exit in ['entry', 'exit']:
    ptdata_fn = f'./prompt_out/{entry_exit}_energy.h5'
    pt_h5file = h5py.File(ptdata_fn)
    # print(list(pt_h5file.keys()))
    outfile = f'{entry_exit}_energy_spectrum.svg'
    outfile = os.path.join(os.path.dirname(__file__),outfile)
    for mon in mcstas_data.keys():
        if mon.startswith(f'eng_{entry_exit}'):
            fig, (intensity,ratio) = plt.subplots(2,1,sharex=True,height_ratios=[2,1])
            plt.xlabel('eV')
            plt.xscale('log')
            ptxx = pt_h5file['center']
            ptyy = pt_h5file['weight']
            intensity.step(ptxx[()] , ptyy[()], linewidth=2,label=f'prompt,{ptyy[()].sum():.2f}')
            mcxx = mcstas_data[mon]['Energy__meV___log_']
            mcyy = mcstas_data[mon]['data']
            intensity.set_yscale('log')
            intensity.plot(10 ** (mcxx[()]-3), mcyy[()],'r+',label=f'mcstas,{mcyy[()].sum():.2f}')
            intensity.legend()

            ratio.hlines(0, ptxx[()].min(), ptxx[()].max(), 'black')
            ratio.step(ptxx[()], (ptyy[()] - mcyy[()])/mcyy[()] * 100, 'g')
            ratio.set_yticks([-100,-50,0,50,100])
            plt.savefig(outfile)

# plot position distribution at exit
for entry_exit in ['entry', 'exit']:
    ptdata_fn = f'./prompt_out/{entry_exit}.h5'
    pt_h5file = h5py.File(ptdata_fn)
    # print(list(pt_h5file.keys()))

    outfile = f'{entry_exit}_width.svg'
    outfile = os.path.join(os.path.dirname(__file__),outfile)
    for mon in mcstas_data.keys():
        if mon.startswith(f'x_mon_{entry_exit}'):
            fig, (intensity,ratio) = plt.subplots(2,1,sharex=True,height_ratios=[2,1])
            plt.xlabel('x(mm)')
            ptxx = pt_h5file['xcenter']
            ptyy = pt_h5file['weight'][()].sum(1)
            plt.xticks(np.linspace(ptxx[()].min(), ptxx[()].max(), 13))
            intensity.step(ptxx[()] , ptyy, linewidth=2,label=f'prompt,{ptyy.sum():.2f}')
            mcxx = mcstas_data[mon]['x__m_']
            mcyy = mcstas_data[mon]['data']
            intensity.set_yscale('log')
            intensity.plot(mcxx[()]*1000, mcyy[()],'r+',label=f'mcstas,{mcyy[()].sum():.2f}')
            intensity.legend()

            ratio.hlines(0, ptxx[()].min(), ptxx[()].max(), 'black')
            ratio.step(ptxx[()], (ptyy - mcyy[()])/mcyy[()] * 100, 'g')
            ratio.set_yticks([-100,-50,0,50,100])
            plt.savefig(outfile)

mc_h5file.close()
pt_h5file.close()