#!/usr/env/bin python

import numpy as np
import os, sys
import pyvista 
# print(sys.executable)

def tutorial_run(fn, numNeu = 100, showInJupyter = True):
    
    import stdout_redirect as rd
    import contextlib

    pyvista.set_plot_theme("paraview")
    # pyvista.global_theme.background = 'royalblue'
    if showInJupyter:
        # configure global theme
        # panel is the only jupyter backend working good
        # a new backend might be avaible in the future,
        # see https://github.com/pyvista/pyvista/issues/3348#issuecomment-1255282449

        # Update in 11/22/2023: a fully-featured backend is ready but still not good enough 
        # for large size data, track https://github.com/pyvista/pyvista/issues/4652
        # TODO
        pyvista.set_jupyter_backend('server')
        # from pyvista.trame.jupyter import elegantly_launch, launch_server
        # host_server = '127.0.0.1'
        # os.environ['TRAME_DEFAULT_HOST'] = host_server
        
        # elegantly_launch(host=host_server)
        # pyvista.global_theme.trame.server_proxy_enabled = True
        # pyvista.global_theme.trame.server_proxy_prefix = f'{host_server}:'


        from Cinema.Interface.Utils import findData
        
        inputfile=findData(f'gdml/{fn}', '.')

        jupyterPromptRedirect(inputfile)

    else:
        promptRedirect(fn, numNeu)


def fullyRedirect(func):
    # redirection of stdout from: 
    # 1.jupyter to terminal; 2.terminal to nothing at file descriptor level;
    # 3.python stdout to nothing
    def inner(*args):
        import stdout_redirect as rd
        import contextlib
        with rd.jupyter2terminal():
            with rd.stdout_redirected(stdout=sys.__stdout__):
                with open(os.devnull, 'w') as f:
                    with contextlib.redirect_stdout(f):
                        func(*args)
    return inner

@fullyRedirect
def promptRedirect(fn, numNeu):
    import subprocess
    subprocess.check_call(['prompt', '-g', fn, '-n', f'{numNeu}', '-v'])


@fullyRedirect
def jupyterPromptRedirect(inputfile):
    from Cinema.Prompt import Launcher, Visualiser
    myLcher=Launcher()
    myLcher.loadGeometry(inputfile)
    v = Visualiser('+', printWorld=False, window_size=[1080, 480])
    for i in range(int(100)):
        myLcher.go(1, recordTrj=True, timer=False, save2Dis=False)
        trj = myLcher.getTrajectory()
        try:
            v.addTrj(trj)
        except ValueError:
            print("skip ValueError in File '/Prompt/scripts/prompt', in <module>, v.addLine(trj)")
    v.plotter.show_bounds()
    v.plotter.show_axes()
    return v.show()


# Monkey patching to overwrite plot style
import pyvista.jupyter.notebook as jp
jp.build_panel_bounds = lambda actor: add_axes(actor)

def add_axes(actor):
    """
    Build a panel bounds actor using the plotter cube_axes_actor.
    """
    bounds = {}

    n_ticks = 5
    if actor.GetXAxisVisibility():
        xmin, xmax = actor.GetXRange()
        bounds['xticker'] = {'ticks': np.linspace(xmin, xmax, n_ticks)}

    if actor.GetYAxisVisibility():
        ymin, ymax = actor.GetYRange()
        bounds['yticker'] = {'ticks': np.linspace(ymin, ymax, n_ticks)}

    if actor.GetZAxisVisibility():
        zmin, zmax = actor.GetZRange()
        bounds['zticker'] = {'ticks': np.linspace(zmin, zmax, n_ticks)}

    bounds['origin'] = [xmin, ymin, zmin]
    bounds['grid_opacity'] = 1.
    bounds['show_grid'] = True
    bounds['digits'] = 1
    bounds['fontsize'] = 14

    return bounds
