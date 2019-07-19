"""
Module for loading BluePyOpt optimized model files
"""

import os
import sciunit
from neuronunit.capabilities import ReceivesSquareCurrent, ProducesMembranePotential, Runnable
from neuron import h
import neo
from quantities import ms
import zipfile
import json
import collections

class CellModel(sciunit.Model,
                         ReceivesSquareCurrent,
                         ProducesMembranePotential,
                         Runnable):
    def __init__(self, model_path=None, model_name=None, run_alerts=False):
        # `model_path` is the path to the model's directory
        if not os.path.isdir(model_path):
            raise IOError("Invalid model path: {}".format(model_path))

        if not model_name:
            file_name = os.path.basename(model_path)
            model_name = file_name.split(".")[0]

        self.model_name = model_name
        self.base_path = model_path
        self.owd = os.getcwd()     # original working directory saved to return later
        self.run_alerts = run_alerts

        self.load_mod_files()
        self.load_cell_hoc()

        # get model template name
        # could also do this via other JSON, but morph.json seems dedicated for template info
        with open(os.path.join(self.base_path, "config", "morph.json")) as morph_file:
            model_template = json.load(morph_file, object_pairs_hook=collections.OrderedDict).keys()[0]

        # access model config info
        with open(os.path.join(self.base_path, "config", "parameters.json")) as params_file:
            params_data = json.load(params_file, object_pairs_hook=collections.OrderedDict)

        # extract v_init and celsius (if available)
        v_init = None
        celsius = None
        try:
            for item in params_data[model_template]["fixed"]["global"]:
                # would have been better if info was stored inside a dict (rather than a list)
                if "v_init" in item:
                    item.remove("v_init")
                    v_init = float(item[0])
                if "celsius" in item:
                    item.remove("celsius")
                    celsius = float(item[0])
        except:
            pass
        if v_init == None:
            h.v_init = -70.0
            print("Could not find model specific info for `v_init`; using default value of {} mV".format(str(h.v_init)))
        else:
            h.v_init = v_init
        if celsius == None:
            h.celsius = 34.0
            print("Could not find model specific info for `celsius`; using default value of {} degrees Celsius".format(str(h.celsius)))
        else:
            h.celsius = celsius

        self.cell = getattr(h, model_template)(os.path.join(str(self.base_path), "morphology"))
        self.iclamp = h.IClamp(0.5, sec=self.cell.soma[0])
        self.vm = h.Vector()
        self.vm.record(self.cell.soma[0](0.5)._ref_v)
        sciunit.Model.__init__(self, name=model_name)

    def load_mod_files(self):
        os.chdir(self.base_path)
        libpath = "x86_64/.libs/libnrnmech.so.0"
        os.system("nrnivmodl mechanisms")   # do nrnivmodl in mechanisms directory
        if not os.path.isfile(os.path.join(self.base_path, libpath)):
            raise IOError("Error in compiling mod files!")
        h.nrn_load_dll(str(libpath))
        os.chdir(self.owd)

    def load_cell_hoc(self):
        with open(os.path.join(self.base_path, self.model_name+'_meta.json')) as meta_file:
            meta_data = json.load(meta_file, object_pairs_hook=collections.OrderedDict)
        best_cell = meta_data["best_cell"]
        self.hocpath = os.path.join(self.base_path,"checkpoints",str(best_cell))

        if os.path.exists(self.hocpath):
            print("Model = {}: using (best cell) {}".format(self.model_name,best_cell))
        else:
            self.hocpath = None
            for filename in os.listdir(os.path.join(self.base_path, "checkpoints")):
                if filename.startswith("cell") and filename.endswith(".hoc"):
                    self.hocpath = os.path.join(self.base_path, "checkpoints", filename)
                    print("Model = {}: cell.hoc not found in /checkpoints; using {}".format(self.model_name,filename))
                    break
            if not os.path.exists(self.hocpath):
                raise IOError("No appropriate .hoc file found in /checkpoints")
        h.load_file(str(self.hocpath))

    def get_membrane_potential(self):
        """Must return a neo.AnalogSignal."""
        signal = neo.AnalogSignal(self.vm,
                                  units="mV",
                                  sampling_period=h.dt * ms)
        return signal

    def inject_current(self, current):
        """
        Injects somatic current into the model.

        Parameters
        ----------
        current : a dictionary like:
                      {'amplitude':-10.0*pq.pA,
                       'delay':100*pq.ms,
                       'duration':500*pq.ms}}
                  where 'pq' is the quantities package
        """
        self.iclamp.amp = current["amplitude"]
        self.iclamp.delay = current["delay"]
        self.iclamp.dur = current["duration"]

    def run(self, tstop):
        t_alert = 100.0
        h.check_simulator()
        h.cvode.active(0)
        self.vm.resize(0)
        h.finitialize(h.v_init)
        while h.t < tstop:
            h.fadvance()
            if self.run_alerts and h.t > t_alert:
                print("\tTime: {} ms out of {} ms".format(t_alert, tstop))
                t_alert += 100.0
