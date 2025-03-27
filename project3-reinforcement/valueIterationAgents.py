# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for _ in range(self.iterations):
            new = self.values.copy()

            # iterate through mdp states
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    continue
                # get possible action values
                actions = self.mdp.getPossibleActions(state)
                # set best value
                bestValue = max([self.getQValue(state, a) for a in actions])
                new[state] = bestValue

            self.values = new



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # initialize q value to be 0
        qValue = 0

        # iterate through every possible outcome of the action
        for next, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            # add reward and future reward*prob of the outcome
            reward = self.mdp.getReward(state, action, next)

            # q value formula
            qValue = qValue + prob * (reward + self.discount * self.values[next])

        return qValue
        # util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        policy = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            policy[action] = self.getQValue(state, action)

        # get the best policy
        return policy.argMax()

        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()

        # start with all 0 values
        for s in states:
            self.values[s] = 0

        numStates = len(states)

        for i in range(self.iterations):
            index = i%numStates
            state = states[index]

            terminal = self.mdp.isTerminal(state)

            if not terminal:
                action = self.getAction(state)
                QValue = self.getQValue(state, action)
                self.values[state] = QValue



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        fringe = util.PriorityQueue()
        predecessors = {}

        for state in states:
            self.values[state] = 0
            predecessors[state] = self.get_predecessors(state)

            terminal = self.mdp.isTerminal(state)

            if not terminal:
                currentValue = self.values[state]
                diff = abs(currentValue - self.maxQ(state))
                fringe.push(state, -diff)





        for _ in range(self.iterations):

            if fringe.isEmpty():
                return

            s = fringe.pop()
            self.values[s] = self.maxQ(s)

            for p in predecessors[s]:
                # find the difference between current value and the highest q value across all possible actions
                diff = abs(self.values[p] - self.maxQ(p))

                if diff > self.theta:
                    fringe.update(p, -diff)

    # define the predecessors of a state
    def get_predecessors(self, s):
        predecessor = set()
        states = self.mdp.getStates()
        movements = ['north', 'south', 'east', 'west']

        # if not terminal state
        if not self.mdp.isTerminal(s):

            for p in states:
                terminal = self.mdp.isTerminal(p)
                legal_actions = self.mdp.getPossibleActions(p)

                if not terminal:
                    for move in movements:
                        if move in legal_actions:
                            transition = self.mdp.getTransitionStatesAndProbs(p, move)

                            for s_prime, T in transition:
                                # no terminal states and T > 0
                                if (s_prime == s) and (T > 0):
                                    predecessor.add(p)

        return predecessor



    # gives the maximum q value
    def maxQ(self, state):
        return max([self.getQValue(state, a) for a in self.mdp.getPossibleActions(state)])









