from quantities.quantity import Quantity
from sciunit import Test
try:
    from sciunit import ObservationError
except:
    from sciunit.errors import ObservationError

import eFELunit.capabilities as cap
from eFELunit import scores
import efel

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import json
# import multiprocessing
import pathos.multiprocessing as multiprocessing
import functools

try:
    #py3
    from time import perf_counter as timer
except:
    #py2
    from time import time as timer

try:
    import pickle as pickle
except:
    import pickle
import gzip

try:
    import copy_reg
except:
    import copyreg

from types import MethodType

def _pickle_method(method):
    func_name = method.__func__.__name__
    obj = method.__self__
    cls = method.__self__.__class__
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)


try:
    copy_reg.pickle(MethodType, _pickle_method, _unpickle_method)
except:
    copyreg.pickle(MethodType, _pickle_method, _unpickle_method)


class eFELfeatureTest(Test):
    """
    Evaluates the specified feature over a range of injected current stimuli.

    Parameters
    ----------
    observation: list
        Specify the target data for comparison
    name: str
        Name of the test
    feature: str
        Specify the feature to be evaluated
    sim_params: dict
        Specify simulation params: stim_delay, stim_duration, tstop
    parallelize: bool
        Whether to parallelize the test evaluation; can create simulator-specific pickling issues 
    force_run : boolean
        If True and the pickle files containing the model's response to the simulation exists, the simulation won't be run again, traces are loaded from the pickle file
    base_directory : str
        Results will be saved here
    show_plot : bool
        If True, plots will be displayed on screen. Default is False.
    """

    def __init__(self,
                 observation = [],
                 name=None,
                 feature = None,
                 sim_params={"stim_delay": 0, "stim_duration": 1000, "tstop": 1000, "dt": 0.01},
                 parallelize=False,
                 force_run=False,
                 base_directory=None,
                 show_plot=False):
        
        self.feature_map = {
            "initial_fi": {"efel_feature": "inv_first_ISI", "units": "Hz", "title": "Initial frequency", "ylabel": "Frequency", "requires": None},
            "final_fi": {"efel_feature": "inv_last_ISI", "units": "Hz", "title": "Final frequency", "ylabel": "Frequency", "requires": None},
            "mean_fi": {"efel_feature": "mean_frequency", "units": "Hz", "title": "Mean frequency", "ylabel": "Frequency", "requires": None},
            "spikecount": {"efel_feature": "Spikecount_stimint", "units": "", "title": "Spike count", "ylabel": "Spike count", "requires": None},
            "time_to_first_spike": {"efel_feature": "time_to_first_spike", "units": "ms", "title": "Time to first spike", "ylabel": "Time", "requires": "Spikecount_stimint>0"},
            "time_to_second_spike": {"efel_feature": "time_to_second_spike", "units": "ms", "title": "Time to second spike", "ylabel": "Time", "requires": "Spikecount_stimint>1"},
            "time_to_last_spike": {"efel_feature": "time_to_last_spike", "units": "ms", "title": "Time to last spike", "ylabel": "Time", "requires": "Spikecount_stimint>0"},
            "AP1_amp": {"efel_feature": "AP1_amp", "units": "mV", "title": "Amplitude of first spike", "ylabel": "Amplitude", "requires": "Spikecount_stimint>0"},
            "AP2_amp": {"efel_feature": "AP2_amp", "units": "mV", "title": "Amplitude of second spike", "ylabel": "Amplitude", "requires": "Spikecount_stimint>1"},
            "APlast_amp": {"efel_feature": "APlast_amp", "units": "mV", "title": "Amplitude of last spike", "ylabel": "Amplitude", "requires": "Spikecount_stimint>0"},
            "AP1_peak": {"efel_feature": "AP1_peak", "units": "mV", "title": "Peak of first spike", "ylabel": "Peak potential", "requires": "Spikecount_stimint>0"},
            "AP2_peak": {"efel_feature": "AP2_peak", "units": "mV", "title": "Peak of second spike", "ylabel": "Peak potential", "requires": "Spikecount_stimint>1"},
            "AP1_width": {"efel_feature": "AP1_width", "units": "ms", "title": "Half-width of first spike", "ylabel": "Half-width", "requires": "Spikecount_stimint>0"},
            "AP2_width": {"efel_feature": "AP2_width", "units": "ms", "title": "Half-width of second spike", "ylabel": "Half-width", "requires": "Spikecount_stimint>1"},
            "APlast_width": {"efel_feature": "APlast_width", "units": "ms", "title": "Half-width of last spike", "ylabel": "Half-width", "requires": "Spikecount_stimint>0"},
            "iv_curve": {"efel_feature": "voltage_deflection_vb_ssse", "units": "mV", "title": "I-V relationship", "ylabel": "Peak amplitude", "requires": None},
        }

        if feature not in self.feature_map.keys():
            raise Exception("feature must be specified! Valid values: {}".format(self.feature_map.keys()))
        self.feature = feature
        self.efel_feature = self.feature_map[feature]["efel_feature"]
        self.units = self.feature_map[feature]["units"]
        self.title = self.feature_map[feature]["title"]
        self.ylabel = self.feature_map[feature]["ylabel"]
        self.feature_requires = self.feature_map[feature]["requires"]

        # set simulation parameters
        self.stim_delay = sim_params["stim_delay"]
        self.stim_duration = sim_params["stim_duration"]
        self.tstop = sim_params["tstop"]
        if "dt" in sim_params.keys():
            self.dt = sim_params["dt"]
        else:
            self.dt = 0.02 # ms
        
        observation = self.format_data(observation, feature)
        if name:
            self.name = name
        else:
           self.name = "Test for {}".format(self.feature)
        Test.__init__(self, observation, self.name)

        if not base_directory:
            base_directory = os.path.join(".", "Results", "eFELfeatureTest", self.name.replace(" ", "_"))
        self.base_directory = base_directory # updated with model name in generate_prediction()     

        self.required_capabilities += (cap.SomaReceivesCurrentProducesMembranePotential,)

        self.parallelize = parallelize
        self.force_run = force_run
        self.show_plot = show_plot

        self.path_temp_data = None # specified later, because model name is needed
        self.logFile = None # specified later, because model name is needed

        self.figures = []

        self.npool = multiprocessing.cpu_count() - 1

        self.description = "Evaluates {} over a range of injected current stimuli".format(self.feature)

    score_type = scores.RMS

    def format_data(self, observation, feature):
        
        # ensure values are Quantity objects
        for idx, entry in enumerate(observation):
          for key, val in entry.items():
                try:
                    assert type(observation[key]) is Quantity
                except Exception as e:
                    quantity_parts = val.split(" ")
                    number = float(quantity_parts[0])
                    units = " ".join(quantity_parts[1:])
                    observation[idx][key] = Quantity(number, units)

        return observation

    def validate_observation(self, observation):

        try:
            assert type(observation) is list
            for entry in observation:
                assert type(entry) is dict
                assert all(key in entry.keys() for key in ["i_inj", "value"])
                for key in entry.keys():
                    assert type(entry[key]) is Quantity
        except Exception as e:
            raise ObservationError(("Observation must be of the form: [{'i_inj':float*pA, 'value':float{}}, ...]"
                                    .format("*"+self.units if self.units else "")))

    def cclamp(self, model, feature_list, amp, delay, dur, tstop):
        if self.base_directory:
            self.path_temp_data = os.path.join(self.base_directory, "temp_data")

        try:
            if not os.path.exists(self.path_temp_data):
                os.makedirs(self.path_temp_data)
        except OSError as e:
            if e.errno != 17:
                raise
            pass

        file_name = os.path.join(self.path_temp_data, 'cclamp_' + str(amp) + '.p')

        if self.force_run or (os.path.isfile(file_name) is False):
            traces=[]
            current = {
                "amplitude": amp,
                "delay": delay,
                "duration": dur,
            }
            trace = model.runsim_stimulus_get_vm_efel_format(tstop, current)
            traces.append(trace)
            traces_results = efel.getFeatureValues(traces, feature_list)
            traces.append(traces_results)
            pickle.dump(traces, gzip.GzipFile(file_name, "wb"))
        else:
            traces = pickle.load(gzip.GzipFile(file_name, "rb"))
        return traces

    def generate_prediction(self, model):
        """Implementation of sciunit.Test.generate_prediction."""

        # update self.base_directory with model name
        self.base_directory = os.path.join(self.base_directory, model.name.replace(" ", "_"))
        self.logFile = os.path.join(self.base_directory, "test_log.txt")

        efel.reset()
        efel.setDoubleSetting('interp_step', self.dt)

        # stimulus levels (current injection) extracted from observation
        amps = [x["i_inj"] for x in self.observation]

        # check if requested feature has prerequisites
        self.feature_list = [self.efel_feature]
        if self.feature_requires:
            self.feature_list.append(self.feature_requires.split(">")[0])
        
        # get the start time
        start_time = timer()

        if self.parallelize:
            pool = multiprocessing.Pool(1, maxtasksperchild=1)
            cclamp_ = functools.partial(self.cclamp, model = model, feature_list = self.feature_list, delay = self.stim_delay, dur = self.stim_duration, tstop = self.tstop)
            results = pool.map(cclamp_, amps, chunksize=1)
            pool.terminate()
            pool.join()
            del pool
        else:
            results = []
            for amp in amps:
                print("I = {}".format(amp))
                results.append(self.cclamp(model = model, feature_list = self.feature_list, amp = amp, delay = self.stim_delay, dur = self.stim_duration, tstop = self.tstop))

        # get the end time
        end_time = timer()

        # model execution time
        self.elapsed_time = end_time - start_time

        # Generate prediction
        prediction = []
        for entry in results:
            if self.feature_requires:
                req_feature_value = entry[1][0][self.feature_requires.split(">")[0]][0]
                if req_feature_value > int(self.feature_requires.split(">")[1]):
                    value = Quantity(round(entry[1][0][self.efel_feature][0], 2), self.units)
                else:
                    value = None
            else:
                if entry[1][0][self.efel_feature]:
                    value = Quantity(round(entry[1][0][self.efel_feature][0], 2), self.units)
                else:
                    value = None
            prediction.append({"i_inj": entry[0]['stim_amp'][0], "value": value})
        sorted(prediction, key=lambda d: d['i_inj'])

        plt.close('all') #needed to avoid overlapping of saved images when the test is run on multiple models in a for loop

        efel.reset()

        # Generate response_stim_X.pdf figures
        plt.figure()
        for entry in results:
            plt.plot(entry[0]['T'], entry[0]['V'])
            plt.title("Response to stimulus = " + str(entry[0]['stim_amp'][0]))
            plt.xlabel("Time (ms)")
            plt.ylabel("Membrane potential (mV)")
            #plt.tick_params(labelsize=18)
            fig_name = "response_stim_" + str(entry[0]['stim_amp'][0]).replace(" ", "") + '.pdf'
            plt.savefig(os.path.join(self.base_directory, fig_name), dpi=600, bbox_inches='tight')
            self.figures.append(os.path.join(self.base_directory, fig_name))
            plt.close('all') 

        return prediction

    def compute_score(self, observation, prediction, verbose=False):
        """Implementation of sciunit.Test.score_prediction."""

        # Following result related files will be generated by this test:
        #   JSON
        #       - compare_obs_pred.json (for each simulated i_inj: obs, pred)
        #       - test_summary.json  (observation, prediction, final score)
        #   Logs
        #       - test_log.txt
        #   Figures
        #       - result_plot.pdf  (parameter vs i_inj relationship plot; model vs target data for each level of i_inj)
        #       - response_stim_X.pdf (see generate_prediction(); multiple Vm vs t plots; one per stimulus level 'X') 

        # Evaluate the score
        score, compare_data, self.total_penalty = scores.RMS.compute(observation, prediction)

        # Generate compare_obs_pred.json
        file_name_sc = os.path.join(self.base_directory, 'compare_obs_pred.json')
        json.dump(compare_data, open(file_name_sc, "w"), default=str, indent=4)

        def remove_dimensionless_units(data_list):
            # to remove dimensionless string for values without units
            # print just value as string (as string for consistency with other data)
            for ind, item in enumerate(data_list):
                if "dimensionless" in str(item["value"]):
                    data_list[ind]["value"] = str(item["value"].magnitude)    
            return data_list

        # Generate test_summary.json
        summary = {
            "observation": remove_dimensionless_units(observation),
            "prediction": remove_dimensionless_units(prediction),
            "score": score,
            "total_applied_penalty": self.total_penalty,
            "model_execution_time": "%0.2f s"%self.elapsed_time
        }
        file_name_summary = os.path.join(self.base_directory, 'test_summary.json')
        json.dump(summary, open(file_name_summary, "w"), default=str, indent=4)

        # Generate result_plot.pdf
        amps = [ float(x["i_inj"]) for x in compare_data ]
        obs = [ float(x["obs"]) for x in compare_data ]
        pred = [ float(x["pred"]) if x["pred"] else None for x in compare_data ]

        plt.figure()
        fig = plt.gcf()
        plt.plot(amps, obs, 'o-r')
        plt.plot(amps, pred, 'x-b')
        plt.title(self.title)
        plt.legend(['Data', 'Model'], loc='best')
        plt.xlabel("$I_{applied} (pA)$")
        ylabel = self.ylabel + " (" + self.units + ")" if self.units else self.ylabel
        plt.ylabel(ylabel)
        fig_name = os.path.join(self.base_directory, "result_plot.pdf")
        plt.savefig(fig_name, dpi=600, bbox_inches='tight')
        self.figures.append(fig_name)
        if self.show_plot:
            plt.show()

        with open(self.logFile, "w") as outfile:
            outfile.write("Overall score: " + str(score) + "\n")
            outfile.write("Model execution time: %0.2f s\n"% self.elapsed_time)
            outfile.write("Total penalty: %d\n"% self.total_penalty)
            outfile.write("------------------------------------------------------------\n")

        return score

    def bind_score(self, score, model, observation, prediction):

        self.figures.append(os.path.join(self.base_directory, 'compare_obs_pred.json'))
        self.figures.append(os.path.join(self.base_directory, 'test_summary.json'))
        self.figures.append(self.logFile)
        score.related_data["figures"] = self.figures
        return score