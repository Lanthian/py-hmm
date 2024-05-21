"""main.py: Provides an HMM dataclass to store and calculate future 
probabilities of a given model. Basic implementation as of current."""

__author__ = "Liam Anthian"


class HMM:
    # Dictionary of probability of transition to a state from prior state
    t_pb: dict[...: dict[...: float]]

    def __init__(self):
        """Constructor method for HMM. Currently fixed."""
        self.t_pb = {}

        # Fixed example for now
        self.t_pb['T'] = {}
        self.t_pb['F'] = {}
        # to state X from state Y
        self.t_pb['T']['T'] = 0.7
        self.t_pb['T']['F'] = 0.3
        self.t_pb['F']['T'] = 0.3 # 1 - self.t_pb['T']['T']
        self.t_pb['F']['F'] = 0.7 # 1 - self.t_pb['T']['F']


    def transition(self, prv: dict[...: float]) -> dict[...: float]:
        """Takes previous probabilities `prv` of states to occur and calculates 
        the next probabilities 1 step ahead, using transition probs `self.t_pb`.
        Returns a dictionary of states and likelihoods."""
        odds = {}
        for p_to in self.t_pb.keys():
            f_pb = self.t_pb[p_to]

            odds[p_to] = 0
            for p_from in f_pb.keys():
                odds[p_to] += f_pb[p_from] * prv[p_from]
        return odds

h = HMM()
print(h.t_pb)

state = {"T": 1, "F": 0}
print(state)
# find converging distribution here
for _ in range(20):
    state = h.transition(state)
    print(state)
