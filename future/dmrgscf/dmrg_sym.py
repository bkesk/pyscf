#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import pyscf.symm

IRREP_MAP = {'D2h': (1,         # Ag
                     4,         # B1g
                     6,         # B2g
                     7,         # B3g
                     8,         # Au
                     5,         # B1u
                     3,         # B2u
                     2),        # B3u
             'C2v': (1,         # A1
                     4,         # A2
                     2,         # B1
                     3),        # B2
             'C2h': (1,         # Ag
                     4,         # Bg
                     2,         # Au
                     3),        # Bu
             'D2' : (1,         # A
                     4,         # B1
                     3,         # B2
                     2),        # B3
             'Cs' : (1,         # A'
                     2),        # A"
             'C2' : (1,         # A
                     2),        # B
             'Ci' : (1,         # Ag
                     2),        # Au
             'C1' : (1,)}

def d2h_subgroup(gpname):
    if gpname.lower() == 'dooh':
        gpname = 'D2h'
    elif gpname.lower() == 'coov':
        gpname = 'C2v'
    else:
        gpname = pyscf.symm.std_symb(gpname)
    return gpname

def irrep_name2id(gpname, symb):
    irrep_id = pyscf.symm.irrep_name2id(gpname, symb) % 10
    gpname = d2h_subgroup(gpname)
    return IRREP_MAP[gpname][irrep_id]

def convert_orbsym(gpname, orbsym):
    '''Convert orbital symmetry irrep_id to Block internal irrep_id
    '''
    if gpname.lower() == 'dooh':
        orbsym = [IRREP_MAP['D2h'][i % 10] for i in orbsym]
    elif gpname.lower() == 'coov':
        orbsym = [IRREP_MAP['C2h'][i % 10] for i in orbsym]
    else:
        gpname = pyscf.symm.std_symb(gpname)
        orbsym = [IRREP_MAP[gpname][i] for i in orbsym]
    return orbsym
