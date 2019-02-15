"""
Tests of the neuron responses to current steps of different amplitudes match experimental data.

The responses are quantified by extracting features from the voltage traces using eFEL.

Reference data (features extracted from experimental recordings) and experimental protocol configurations
 are extracted from .zip files produced by BluePyOpt.

Andrew Davison and Shailesh Appukuttan, UNIC, CNRS.
March 2017
"""

import os.path
from datetime import datetime
import json
import sciunit
from neuronunit.capabilities import ProducesMembranePotential, ReceivesSquareCurrent
import neo
import efel
import matplotlib.pyplot as plt
from quantities import ms
from eFELunit.scores import RMS_ZScore

class MultipleCurrentStepTest(sciunit.Test):
    """
    Tests of the neuron responses to current steps of different amplitudes match
    experimental data.

    The responses are quantified by extracting features from the voltage traces
    using eFEL.
    """
    required_capabilities = (ProducesMembranePotential, ReceivesSquareCurrent)
    score_type = RMS_ZScore

    def __init__(self, observation=None, name=None, protocol=None, plot_figure=False):
        sciunit.Test.__init__(self, observation, name)
        self.plot_figure = plot_figure
        if protocol is None:
            raise ValueError("Must provide a stimulation protocol")
        self.protocol = protocol
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.figures = []

    def validate_observation(self, observation):
        """
        Checks that the observation has the correct format, i.e.

        - a top-level dict with one entry per current step.
            - the key should be a label for the step
            - the value should be a dict containing one entry per feature of the voltage trace
                - the key of the feature dict should be a label for the feature
                - the value should be a dict with keys 'mean' and 'value'
        """
        pass   # todo

    def generate_prediction(self, model, verbose=False):
        use_cache = True
        self.test_base_dir = os.path.join(model.base_path, "validations", self.name)
        if not os.path.isdir(self.test_base_dir):
            os.makedirs(self.test_base_dir)
        cache_filename = os.path.join(self.test_base_dir, "results.pkl")
        if use_cache and os.path.exists(cache_filename):
            print("***** Using cache to retrieve relevant model data *****")
            io = neo.io.get_io(cache_filename)
            self.recordings = io.read_block()
        else:
            print("Note: no cached data for this model specification!")
            self.recordings = self._run_simulations(model)
            io = neo.io.PickleIO(cache_filename)
            io.write_block(self.recordings)
        if self.plot_figure:
            for i, seg in enumerate(self.recordings.segments):
                plt.plot(seg.analogsignals[0].times.rescale('ms'),
                         seg.analogsignals[0].rescale('mV').magnitude + i * 110.0,
                         label=seg.name)
            plt.legend()
            self.figure_path = os.path.join(self.test_base_dir, "{}_{}.png".format(model.name, self.timestamp))
            plt.savefig(self.figure_path)
            self.figures.append(self.figure_path)
        return self._calculate_features(self.recordings)

    def _run_simulations(self, model):
        """For each step in the protocol, run simulation and store recordings"""
        recordings = neo.Block()
        print("Total protocols: {}".format(len(self.protocol)))
        for idx, item in enumerate(self.protocol.items()):
            step_name = item[0]
            step = item[1]
            segment = neo.Segment(name=step_name)
            recordings.segments.append(segment)
            segment.block = recordings

            print("{}. Current protocol: {}".format(idx+1, step_name))
            model.inject_current(step["stimuli"])
            model.run(tstop=step["total_duration"])
            signal = model.get_membrane_potential()
            stimulus_on =  neo.Epoch(times=step["stimuli"]["delay"]*ms,
                                     durations=step["stimuli"]["duration"]*ms,
                                     labels="stimulus")
            segment.analogsignals.append(signal)
            segment.epochs.append(stimulus_on)
        return recordings

    def _calculate_features(self, recordings):
        """For each recorded step, calculate the features."""
        features_from_simulation = {}
        for segment in recordings.segments:
            step_name = segment.name
            feature_names = self.observation[step_name].keys()
            trace = {
                'T': segment.analogsignals[0].times.rescale('ms').magnitude,
                'V': segment.analogsignals[0].rescale('mV').magnitude,
                'stim_start': [segment.epochs[0].times],
                'stim_end': [segment.epochs[0].times + segment.epochs[0].durations]
            }

            features = efel.getFeatureValues([trace], feature_names)[0]
            features_from_simulation[step_name] = dict([(k, {'value': v[0]})
                                                        for k, v in features.items()])
        return features_from_simulation

    def compute_score(self, observation, prediction, verbose=False):
        """
        Generates a score given the observations provided in the constructor
        and the prediction generated by generate_prediction.
        """
        # reformat the observations and predictions into the form needed by RMS_ZScore
        # i.e. from dict of dicts into a flat list of dicts
        observation_list = []
        prediction_list = []
        for step_name in observation:
            for feature_name in observation[step_name]:
                key = "{}_{}".format(step_name, feature_name)
                observation_list.append({key: observation[step_name][feature_name]})
                prediction_list.append({key: prediction[step_name][feature_name]})
        return self.score_type.compute(observation_list, prediction_list)

    def bind_score(self, score, model, observation,
                   prediction):
        """
        For the user to bind additional features to the score.
        """
        if hasattr(self, "figure_path"):
            score.related_data["figures"] = self.figures

        for key, val in score.__dict__["model"].__dict__.items():
            if val.__class__.__name__ == "HocObject":
                score.__dict__["model"].__dict__.pop(key)
        
        return score



class BluePyOpt_MultipleCurrentStepTest(MultipleCurrentStepTest):
    """
    Tests if the neuron responses to current steps of different amplitudes match
    experimental data.

    The responses are quantified by extracting features from the voltage traces
    using eFEL.

    Experimental protocol definitions and experimental features obtained from
    zip files produced by BluePyOpt
    """

    def __init__(self, observation_dir=None, name=None, plot_figure=False, **test_kwargs):
        # This test does not employ external observation data, but is
        # rather located within each model's directory. So test results for
        # various models are not necessarily judged against the same data.

        # extract model specific observation data


        # load the protocol definition and the reference data
        with open(os.path.join(observation_dir, "config", "protocols.json")) as fp:
            protocols = json.load(fp)
            assert len(protocols) == 1
            template_name = protocols.keys()[0]

        with open(os.path.join(observation_dir, "config", "features.json")) as fp:
            reference_features = json.load(fp)
        assert reference_features.keys()[0] == template_name

        # reformat the reference_features dict into the necessary form
        observations = {}
        for step, value in reference_features[template_name].items():
            observations[step] = {}
            for feature_name, (mean, std) in value["soma"].items():
                observations[step][feature_name] = {"mean": mean, "std": std}

        # reformat the protocol definition into the form requested by NeuronUnit
        protocol = {}
        for step_name, content in protocols[template_name].items():
            stim = content["stimuli"][0]
            stim["amplitude"] = stim["amp"]
            protocol[step_name] = {
                "stimuli": stim,
                "total_duration": stim["totduration"]
            }
            del stim["amp"]
            del stim["totduration"]

        MultipleCurrentStepTest.__init__(self,
                                         observation=observations,
                                         name=name,
                                         protocol=protocol,
                                         plot_figure=plot_figure)
