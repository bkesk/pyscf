#!/usr/bin/env python

'''
GHF with PBC-SOC-ECP
'''

import numpy as np

from pyscf import lib

from pyscf.pbc import scf as scf_pbc
from pyscf.pbc import gto as gto_pbc

ecp_soc = {'Bi' : '''
Bi nelec 60
Bi ul
2       1.0000000              0.0000000
Bi S
2      13.0430900            283.2642270
2       8.2216820             62.4719590
Bi P
2      10.4677770             72.0014990     -144.002998
2       9.1189010            144.0022770      144.002277
2       6.7547910              5.0079450      -10.015890
2       6.2525920              9.9915500        9.991550
Bi D
2       8.0814740             36.3962590      -36.396259
2       7.8905950             54.5976640       36.398443
2       4.9555560              9.9842940       -9.984294 
2       4.7045590             14.9814850        9.987657
Bi F
2       4.2145460             13.7133830       -9.142256
2       4.1334000             18.1943080        9.097154
Bi G
2       6.2057090            -10.2474430        5.123722
2       6.2277820            -12.9557100       -5.182284
'''}


cell = gto_pbc.Cell()
cell.atom = [['Bi',(2.772689540484, 0.0, 0.922719929252)],
             ['Bi',(7.653699135047, 0.0, 2.547065086548)]]
cell.a = [[7.295032054, -3.989777232, 0.000000000],
          [7.295032054, 3.989777232, 0.000000000],
          [5.112954957, 0.000000000, 6.556943391]]
cell.precision = 1e-5
cell.exp_to_discard = 0.1
cell.basis = 'ccpvdzpp'
cell.ecp = ecp_soc

cell.verbose=4

cell.build()

ghf = scf_pbc.GHF(cell).density_fit()
E = ghf.kernel()

ghf.with_soc = True
Esoc = ghf.kernel()

print(f" Esoc - E = {Esoc-E}")

# Adding on-site SOC by hand for comparison:
#   not expected to give correct results with small
#   separations, but should give correct results
#   for large separation and Gamma-point.

ghf = scf_pbc.GHF(cell).density_fit()

s = .5 *lib.PauliMatrices
ecpso = np.einsum('sxy,spq->xpyq', -1j * s, cell.intor('ECPso'))
hcore = ghf.get_hcore()
hcore = hcore + ecpso.reshape(hcore.shape)
ghf.get_hcore = lambda *args: hcore

Esoc_onsite = ghf.kernel()

print(f" E SOC - E SOC (on-site) = {Esoc-Esoc_onsite}")
