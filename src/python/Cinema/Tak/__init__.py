from Cinema.Interface import *

_tak_cal_limit = importFunc('tak_cal_limit', None, [type_dbl, type_dbl, type_npdbl1d,  type_npdbl1d, type_int,  type_npdbl1d, type_npdbl1d, type_npdbl1d, type_npdbl1d] )
_tak_cal_integral = importFunc('tak_cal_integral', None, [type_dbl, type_dbl, type_int,  type_npdbl1d,  type_npdbl1d, type_int, type_npdbl1d, type_npdbl1d, type_npdbl1d, type_npdbl1d] )

def tak_cal_limit_integral(x, y, tarr, mass, temperature):
    if x.size != y.size:
        raise RuntimeError('x.size != y.size')
    l_cls=np.zeros(tarr.size)
    l_real=np.zeros(tarr.size)
    l_imag=np.zeros(tarr.size)
    _tak_cal_limit(mass,temperature,x,y,tarr.size,tarr,l_cls,l_real,l_imag)
    return l_cls,l_real,l_imag

def tak_cal_filon(x,y,tarr,mass, temperature):
    if x.size != y.size:
        raise RuntimeError('x.size != y.size')

    # omega size should be odd when use filon
    if x.size%2!=1:
        x=x[:-1]
        y=y[:-1]
        raise RuntimeError('x should be odd')

    panels = (x.size-1)//2
    i_cls=np.zeros(tarr.size)
    i_real=np.zeros(tarr.size)
    i_imag=np.zeros(tarr.size)
    _tak_cal_integral(mass, temperature,panels,x,y,tarr.size,tarr,i_cls,i_real,i_imag)
    return i_cls, i_real, i_imag
