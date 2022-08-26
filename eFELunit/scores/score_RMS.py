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
        for obs, pred in zip(observation, prediction):
            sum_scores = sum_scores + abs(pow(float(obs["value"]),2) - pow(float(pred["value"]),2))

            compare_data.append({
                "i_inj": obs["i_inj"],
                "obs": obs["value"] if "dimensionless" not in str(obs["value"]) else obs["value"].magnitude,
                "pred": pred["value"] if "dimensionless" not in str(pred["value"]) else pred["value"].magnitude
            })
        RMS_score = pow((sum_scores/len(observation)), 1/2)
        return RMS(RMS_score), compare_data

    def __str__(self):
        return '%.2f' % self.score