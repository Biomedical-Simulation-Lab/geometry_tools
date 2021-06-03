#///////////////////////////////////////////////////////////////
#// CFD job script wrapper
#// Copyright (C) 2018 Mehdi Najafi (mnuoft@gmail.com)
#// Distribution of this file is not allowed in any form.
#///////////////////////////////////////////////////////////////

class SubmissionTemplate():
    def __init__(self, case_name, cycles=2, timesteps_per_cycle=9600,
                uOrder=1, save_frequency=10, estimated_required_time='24:00:00', 
                post_processing_time_minutes='180', num_cores=80):

        self.case_name = case_name
        self.template = {
            '#!/bin/bash' : '',
            '# CFD job script wrapper' : '',
            'casename=' : '"{}"'.format(case_name),
            'cycles=' : '{}'.format(cycles),
            'timesteps_per_cycle=' : '{}'.format(timesteps_per_cycle),
            'uOrder=' : '{}'.format(uOrder),
            'save_frequency=' : '{}'.format(save_frequency),
            'estimated_required_time=' : '"{}"'.format(estimated_required_time),
            'post_processing_time_minutes=' : '{}'.format(post_processing_time_minutes),
            'num_cores=' : '{}'.format(num_cores),
            '~/../mnajafi/solver.bin $0 "$@"' : '',
        }

    def save_script(self, save_dir):
        keys = self.template.keys()
        values = [self.template[k] for k in keys]
        lines = []
        for k,  v in zip(keys, values):
            lines.append(''.join([k,v]))
        
        save_file = save_dir / (self.case_name + '.sh')
        textfile = open(save_file, "w")
        for element in lines:
            textfile.write(element + "\n")
        textfile.close()