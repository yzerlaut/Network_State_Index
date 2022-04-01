import os
import shutil

import numpy as np
import pandas as pd

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

data_directory = os.path.join(os.path.expanduser('~'), 'Downloads', 'ecephys_cache_dir')

manifest_path = os.path.join(data_directory, "manifest.json")

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

all_sessions = cache.get_session_table() # get all sessions

sessions = all_sessions[(all_sessions.sex == 'M') & \
                        (all_sessions.full_genotype.str.find('wt/wt') > -1) & \
                        (all_sessions.session_type == 'functional_connectivity') & \
                        (['VISp' in acronyms for acronyms in 
                               all_sessions.ecephys_structure_acronyms])]
print('Number of sessions with the desired characteristics: ' + str(len(sessions)))
sessions.head()

for session_id, row in sessions.iterrows():

    truncated_file = True
    directory = os.path.join(data_directory + '/session_' + str(session_id))
    
    while truncated_file:
        
        session = cache.get_session_data(session_id)

        pupil = session.get_pupil_data()
        
        try:
            print(session.specimen_name)
            truncated_file = False
        except OSError:
            shutil.rmtree(directory)
            print(" Truncated spikes file, re-downloading")

        
        for probe_id, probe in session.probes.iterrows():

            if probe.description=='probeC':

                truncated_lfp = True

                while truncated_lfp:
                    try:
                        lfp = session.get_lfp(probe_id)
                        truncated_lfp = False
                    except OSError:
                        fname = directory + '/probe_' + str(probe_id) + '_lfp.nwb'
                        os.remove(fname)
                        print("  Truncated LFP file, re-downloading")
                    except ValueError:
                        print("  LFP file not found.")
                        truncated_lfp = False



