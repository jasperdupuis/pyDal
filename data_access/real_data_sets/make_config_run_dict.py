# -*- coding: utf-8 -*-
"""

Needs list_runs parameters fed to this module.

"""

import pickle

fname = r'config_speed_run_dictionary.pkl'

if __name__ == '__main__':
    configs = []
    speeds = []
    headings = [] 
    for r in list_runs:
        configs.append(r[8])
        speeds.append(r[6:8])
        headings.append(r[-2])
    configs = set(configs) ; speeds.remove('00') # Make set, but remove ambient
    speeds = set(speeds) ; configs.remove('X') # Make set, but remove ambient
    headings = set(headings) ; headings.remove('X') # Make set, but remove ambient
    
    result = dict()
    for c in configs:
        for s in speeds:
            local_result = []
            for r in list_runs:
                if r[8] == c\
                and r[6:8] == s:
                    local_result.append(r)            
            if not ( len ( local_result ) == 0 ): # if empty set skip entering
                key= c + '_' + s
                result[key] = local_result
    
    
    with open( fname, 'wb' ) as file:
        pickle.dump( result, file )
