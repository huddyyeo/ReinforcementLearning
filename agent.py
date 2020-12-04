############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################
import time
import numpy as np
import torch
import collections


class ReplayBuffer:
    def __init__(self, maxlen=5000):
        # contains a deque for the replay buffer
        # contains also a deque for the weights and an attribute for the probabilities for prioritised action replay
        # the probabilies are computed right before each time a batch is sampled
        self.buffer = collections.deque(maxlen=maxlen)
        self.p = None
        self.w = collections.deque(maxlen=maxlen)


class duelNet(torch.nn.Module):
    def __init__(self, input_dimension, dim_1=500):

        super(duelNet, self).__init__()

        self.layer_in = torch.nn.Linear(in_features=input_dimension, out_features=dim_1)

        self.layer_v = torch.nn.Linear(in_features=dim_1, out_features=1)
        self.layer_a = torch.nn.Linear(in_features=dim_1, out_features=4)

        self.relu = torch.nn.ReLU()

    def forward(self, inp):
        # implementing the duelling deep Q architecture proposed by DeepMind

        # first hidden layer
        out = self.relu(self.layer_in(inp))

        # calculate value stream, returns 1 value for state
        values = self.layer_v(out)

        # calculate advantage stream, returns 4 advantages, 1 for each action
        adv = self.layer_a(out)

        # calculate mean of the advantage
        adv_mean = torch.mean(adv.unsqueeze(-1), axis=1)

        # aggregating layer
        q = values + adv.squeeze() - adv_mean

        return q


class Agent:

    # Function to initialise the agent
    def __init__(self, lr=0.005, duel=True):
        # self.loss=[]
        self.s_time = time.time()
        # Set the episode length
        self.episode_length = 500
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        self.b = ReplayBuffer(maxlen=5000)

        self.net = duelNet(2)
        self.target = duelNet(2)

        self.update_target()
        self.lr = lr
        self.e = 1
        self.optimiser = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.MSELoss()
        self.stopped = False
        self.test = False
        self.decay = 0.97

        self.net.train()

        self.move_dict = {
            0: np.array([0.02, 0], dtype=np.float32),
            1: np.array([0, 0.02], dtype=np.float32),
            2: np.array([-0.02, 0], dtype=np.float32),
            3: np.array([0, -0.02], dtype=np.float32),
        }

        # no variables relating to past actions or states are stored, other than the buffer

    def update_target(self):
        self.target.load_state_dict(self.net.state_dict())

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken == self.episode_length or (
            self.test and self.num_steps_taken == 100
        ):

            # reset num steps
            self.num_steps_taken = 0

            # if we were testing and did not stop
            if self.test and not self.stopped:
                self.test = False

            # if we were not testing at all
            elif (
                not self.test
                and (time.time() - self.s_time) > 60
                and self.episode_length % 10 == 0
            ):
                # set the testing flag to true
                self.test = True
                return True
            if not self.stopped:

                # decay learning rate and epsilon, update target network and reduce episode length
                self.lr *= self.decay
                self.optimiser = torch.optim.Adam(self.net.parameters(), lr=self.lr)
                self.update_target()

                self.e *= self.decay

                self.episode_length = max(self.episode_length - 5, 100)

                # if we havent found goal after a long time, update some parameters to increase exploration
                if self.e < 0.1:
                    # print('e is low',self.e,time.time()-self.s_time)
                    self.e = 0.45
                    self.decay = 0.98
                    self.episode_length = 525

                    # increase learning rate as well so weights wont stagnate
                    self.lr = 0.002
                    self.optimiser = torch.optim.Adam(self.net.parameters(), lr=self.lr)

            return True

        else:
            return False

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):

        if self.test or self.stopped:
            action = self.get_greedy_action(state, discrete=True)

        elif np.random.random() < (self.e * (1.003 ** self.num_steps_taken)):
            # exploration grows with number of steps per episode
            action = np.random.choice([0, 1, 2, 3], p=[0.3, 0.325, 0.05, 0.325])
            # actions biased against the left, which corresponds to action value 2

        else:
            action = self.get_greedy_action(state, discrete=True)

        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Store the action; this will be used later, when storing the transition
        self.action = action
        return self._discrete_action_to_continuous(action)

    def _discrete_action_to_continuous(self, discrete_action):
        # converts discrete value to continuous, using a dictionary of defined movements
        return self.move_dict[discrete_action]

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):

        # once we have completed early stopping we do not train anymore
        if self.stopped:
            return

        # if we are testing and we hit the goal, set self.stopped to True
        if self.test == True:
            # environment.show(state)
            if distance_to_goal < 0.03:
                self.stopped = True  # once self.stopped is true, all training stops
                self.e = 0
                print(
                    "successful policy! stopped training at",
                    np.round(time.time() - self.s_time, 3),
                )
                self.episode_length = 100
                return

        # Convert the distance to a reward, weight also how far right the agent has moved
        reward = 0.8 * (1 - distance_to_goal) + 0.2 * next_state[0]

        # reducing reward for bumping into wall
        if (
            np.sum((self.state - next_state) ** 2) < 0.00005
        ):  # using a distance metric in case of any rounding errors
            reward -= 0.02
        # since our distance travelled per step is 0.02, we negate the the max expected gain in reward
        # prevents over-generalisation over long periods of moving right and when suddenly there is a wall on the right

        # Create a transition
        transition = (self.state, self.action, reward, next_state)

        self.b.buffer.append(np.array(transition, dtype=object))
        # add max weight otherwise add a min of 0.01
        if len(self.b.w) != 0:

            w = np.max(self.b.w)
            self.b.w.append(w)
        else:
            self.b.w.append(0.01)

        # dont train if we're testing
        if self.test:
            return

        # batch sizes of 50 are used here
        if len(self.b.buffer) > 50:
            self.rebalance_p()
            minibatch_indices = np.random.choice(
                range(len(self.b.buffer)), 50, p=self.b.p, replace=False
            )
            train_batch = np.vstack([self.b.buffer[i] for i in minibatch_indices])
            loss = self.train_batch(train_batch, minibatch_indices)  # train the network

    def train_batch(self, batch, batch_indices, gamma=0.90):
        # set net to train
        self.net.train()
        state, action, reward, next_state = [
            torch.tensor(np.vstack(batch[:, i])).squeeze() for i in range(4)
        ]
        reward = reward.float()

        # obtain actions from target
        tn_actions = torch.argmax(self.target(next_state).detach(), axis=1)

        # obtain q values from q net using actions from target
        q2 = self.net(next_state)
        q2 = torch.gather(q2, dim=1, index=tn_actions.unsqueeze(-1)).squeeze(-1)

        # combine with reward and gamma
        q2 = reward + gamma * q2

        q1 = self.net(state)
        q1 = torch.gather(q1, dim=1, index=action.unsqueeze(-1)).squeeze(-1)

        self.optimiser.zero_grad()

        # calculate the loss and clamp between -1,1 to ensure stability
        loss = torch.clamp(self.loss_fn(q2, q1), -1, 1)

        loss.backward()
        self.optimiser.step()

        # update weights for prioritised action replay
        new_w = np.abs((q2 - q1).detach().numpy()) + 0.01

        # convert weight deque to np array, update the weights for batch indices, then convert back to a deque
        self.b.w = np.array(self.b.w)
        self.b.w[batch_indices] = new_w
        self.b.w = collections.deque(self.b.w, maxlen=5000)

        return loss.item()

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state, discrete=False):
        # calculate q values from q net and take argmax
        self.net.eval()
        action = self.net(torch.tensor(state).float().unsqueeze(0)).detach().numpy()
        action = np.argmax(action)
        if discrete:
            return action
        else:
            return self._discrete_action_to_continuous(action)

    def rebalance_p(self, a=0.7):
        # updating p values based on current weights
        w_power = np.array(self.b.w) ** a
        self.b.p = (w_power / np.sum(w_power)).tolist()
