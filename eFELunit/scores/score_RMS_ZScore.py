from sciunit.scores import ZScore
import quantities
import numpy as np

#==============================================================================

class RMS_ZScore(ZScore):
    """
	Calculates the z-score for one or more variables and returns the root mean square
    """

    _allowed_types = (float,)

    _description = ('The root-mean-square of the z-scores for multiple variables')

    @classmethod
    def compute(cls, observation, prediction):
        """
        Computes a z-score from an observation and a prediction.
        """
        scores = []
        table = []  # store intermediate results
        for obs, pred in zip(observation, prediction):
            assert obs.keys() == pred.keys()
            assert len(obs) == 1
            #print("O" + str(obs))
            #print("P" + str(pred))
            scores.append(ZScore.compute(list(obs.values())[0],
                                         list(pred.values())[0]).score)
            #print("Z" + str(scores[-1]) + "\n")
            key = obs.keys()[0]
            table.append((key, obs[key]["mean"], obs[key]["std"], pred[key]["value"], scores[-1]))
        sc = np.sqrt(np.mean(np.square(scores)))
        print("RMS(Z) " + str(sc))
        score = cls(sc, related_data={'score_table': table})
        return score
