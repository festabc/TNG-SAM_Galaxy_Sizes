import numpy as np

#fitting functions for MMW
def f_c(c):
    return 2.0 / 3.0 + (c / 21.5)**0.7

def f_R(lprime, md, c):
    p = -0.06 + 2.71 * md + 0.0047 / lprime;
    return (lprime/0.1)**p * (1.0-3.0*md+5.2*md*md)*(1.0-0.019*c+0.00025*c*c + 0.52/c)

def disk_infall(m_halo,r_vir,conc,spin,mstar_disk,mass_cold_gas):
    ''' disk infall'''
    if conc < 2.0: 
        conc = 2.0
    if conc > 30.0: 
        conc = 30.0
    if spin < 0.02:
        spin=0.02
    r_iso = spin*r_vir/np.sqrt(2.0)   
    mdisk = (mstar_disk + mass_cold_gas)
    fdisk = mdisk/m_halo
    fj=1.0
    if fdisk > 0.02:
        r_disk = fj*r_iso*f_R(fj*spin, fdisk, conc)/np.sqrt(f_c(conc))
    else:
        r_disk = r_iso
    return r_disk

if __name__=='__main__':
    print(disk_infall(1.e12,200,12,0.1,6.e10,1.e10))