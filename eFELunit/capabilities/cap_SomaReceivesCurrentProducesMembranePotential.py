import sciunit

class SomaReceivesCurrentProducesMembranePotential(sciunit.Capability):
    """Enables recording membrane potential from soma """

    def get_soma_vm(self, tstop):
        """
        Run simulation for time 'tstop', specified in ms, while 
        recording the somatic membrane potential.
        Must return a dict of the form:
        	{'T': list1, 'V': list2 } where,
        		list1 = time series (in ms)
        		list2 = membrane potential series (in mV)
        """
        raise NotImplementedError()

    def inject_soma_square_current(self, current):
        """
        Input current is specified in the form of a dict with keys:
            'delay'     : (value in ms),
            'duration'  : (value in ms),
            'amplitude' : (value in nA)
        """
        raise NotImplementedError()

    def reset_model(self):
        raise NotImplementedError()
        
    def runsim_stimulus_get_vm_efel_format(self, tstop, current):
        self.reset_model()
        self.inject_soma_square_current(current)
        traces = self.get_soma_vm(tstop)
        efel_trace = {'T' : traces['T'],
                      'V' : traces['V'],
                      'stim_amp' : [current["amplitude"]],
                      'stim_start' : [current["delay"]],
                      'stim_end'   : [current["delay"] + current["duration"]]}
        return efel_trace