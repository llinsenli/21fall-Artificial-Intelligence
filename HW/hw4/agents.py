import random

# 1. Q-Learning
class QLearningAgent:
    """Implement Q Reinforcement Learning Agent using Q-table."""

    def __init__(self, game, discount, learning_rate, explore_prob):
        """Store any needed parameters into the agent object.
        Initialize Q-table.
        """
        self.game = game
        self.discount = float(discount)
        self.learning_rate = float(learning_rate)
        self.explore_prob = float(explore_prob)
        self.QValues = {}


    def get_q_value(self, state, action):
        """Retrieve Q-value from Q-table.
        For an never seen (s,a) pair, the Q-value is by default 0.
        """
        if (state, action) not in self.QValues.keys():
            self.QValues[(state, action)] = 0
            return self.QValues[(state, action)]
        else:
            return self.QValues[(state, action)]

    def get_value(self, state):
        """Compute state value from Q-values using Bellman Equation.
        V(s) = max_a Q(s,a)
        """
        values = [self.get_q_value(state, action) for action in self.game.get_actions(state)]
        if (values):
            return max(values)
        else:
            return 0.0

    def get_best_policy(self, state):
        """Compute the best action to take in the state using Policy Extraction.
        π(s) = argmax_a Q(s,a)

        If there are ties, return a random one for better performance.
        Hint: use random.choice().
        """
        legal_actions = self.game.get_actions(state) #all the legal actions
        if legal_actions:
            value = self.get_value(state)
            tie = []
            for a in legal_actions:
                if self.get_q_value(state,a)==value:
                    tie.append(a)
            return random.choice(tie)
        return None


    def update(self, state, action, next_state, reward):
        """Update Q-values using running average.
        Q(s,a) = (1 - α) Q(s,a) + α (R + γ V(s'))
        Where α is the learning rate, and γ is the discount.

        Note: You should not call this function in your code.
        """
        newQValue = (1 - self.learning_rate) * self.get_q_value(state, action) #new Qvalue
        #next_action = self.get_best_policy(next_state)
        #newQValue += self.learning_rate * (reward + (self.discount * self.get_q_value(next_state,next_action)))
        newQValue += self.learning_rate * (reward + (self.discount * self.get_value(next_state)))
        self.QValues[(state, action)] = newQValue

    # 2. Epsilon Greedy
    def get_action(self, state):
        """Compute the action to take for the agent, incorporating exploration.
        That is, with probability ε, act randomly.
        Otherwise, act according to the best policy.

        Hint: use random.random() < ε to check if exploration is needed.
        """
        if random.random() < self.explore_prob:
            return random.choice(list(self.game.get_actions(state)))
        return self.get_best_policy(state)
