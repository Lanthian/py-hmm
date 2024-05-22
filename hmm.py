"""hmm.py: Provides HMM and State dataclasses to store and calculate future 
state probabilities of a given model. Basic implementation as of current."""

__author__ = "Liam Anthian"


class HMM:
    # Dictionary of probability of transition to a state from prior state
    t_pb: dict[...: dict[...: float]]

    def __init__(self, trans):
        """Constructor method for HMM."""
        self.t_pb = trans


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


class State:
    """Hashable, equatable, state in an HMM - contains transition probabilities
    from adjacently linked states."""
    id: str                                 # Name of state
    likelihood: dict['State': float]        # Pr(self|other)


    def __init__(self, id, givens: dict['State': float] = {}):
        """Constructor method for State. Transition probability dict `givens` is 
        left blank by default - no linked states."""
        self.id = id
        # Need to copy `givens` to avoid assigning all new states same dict
        self.likelihood = givens.copy()

    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, State):
            return self.id == other.id
        return NotImplemented
    
    def __str__(self):
        return f"{self.id} : {[(k.id,v) for k,v in self.likelihood.items()]}"
    
    
    def add(self, givens: dict['State': float]):
        """Add new / update old transition probabilities from `givens`."""
        for k,pb in givens.items():
            self.likelihood[k] = pb


    def given(self, prv: 'State', prior: float = 1) -> float:
        """Find the probability of a next state given a prior. Prior set as 1 by
        default - assumes guaranteed previous state if not told otherwise.
            Pr(self|other) * Pr(other)"""
        # Error check: Check if no way link between states        
        if prv not in self.likelihood: return 0
        
        # Otherwise return probability from state
        return self.likelihood[prv] * prior
    
    def total(self, priors: dict['State': float]) -> float:
        """Find the total probability of a next state given all priors.
        Returns Σ P(Rt|Rt-1) * P(Rt-1|...)
            (sum of (likelihood times relevant prior))
        """
        return sum([self.given(prv, pb) for prv,pb in priors.items()])
    
    def posterior_x_evidence(self, priors, e_likelihood) -> float | None:
        """Finds partial posterior of a next state given priors & observed data.
        Returns P(Rt|E1:t) * P(E1:t) by calculating P(E1:t|Rt) * P(Rt)
            Normalisation still required with this output.
        Alternatively, returns None if state independent from observed data."""
        # Error check: Check if independent regarding evidence (marginal)
        #   (do not include in calculation if so)
        if self not in e_likelihood: return None # self.total(priors)

        # Otherwise return according to filtering equation
        return e_likelihood[self] * self.total(priors)
