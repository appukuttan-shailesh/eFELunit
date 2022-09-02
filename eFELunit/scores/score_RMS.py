from __future__ import division
from sciunit import Score
import math

class RMS(Score):
    """
    Root mean square between values in observation and prediction
    """
  
    @classmethod
    def compute(cls, observation, prediction):
        """Computes RMS value from an observation and a prediction."""

        compare_data = []
        sum_scores = 0
        penalty = 10
        total_applied_penalty = 0
        for obs, pred in zip(observation, prediction):
            if pred["value"] != None:
                sum_scores = sum_scores + pow(float(pred["value"]) - float(obs["value"]),2)
            else:
                total_applied_penalty = total_applied_penalty + penalty
                sum_scores = sum_scores + penalty # penalty for no prediction

            compare_data.append({
                "i_inj": obs["i_inj"],
                "obs": obs["value"] if "dimensionless" not in str(obs["value"]) else str(obs["value"].magnitude),
                "pred": pred["value"] if "dimensionless" not in str(pred["value"]) else str(pred["value"].magnitude)
            })
        RMS_score = pow((sum_scores/len(observation)), 1/2)
        return RMS(RMS_score), compare_data, total_applied_penalty

    def __str__(self):
        return '%.2f' % self.score