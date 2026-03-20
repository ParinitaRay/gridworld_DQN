import random
import torch
import torch.nn as nn
from collections import deque, namedtuple

from model import DQN
from config import *

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Agent:
    def __init__(self, num_states, num_actions, device):
        self.num_actions = num_actions
        self.device = device
        self.epsilon = EPSILON_START

        self.policy_net = DQN(num_states, HIDDEN_NODES, num_actions).to(device)
        self.target_net = DQN(num_states, HIDDEN_NODES, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.loss_fn = nn.MSELoss()
        self.step_count = 0

    def state_to_tensor(self, state):
        """Convert state index to one-hot tensor"""
        tensor = torch.zeros(NUM_STATES, device=self.device)
        tensor[state] = 1.0
        return tensor

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            with torch.no_grad():
                return self.policy_net(self.state_to_tensor(state)).argmax().item()

    def optimize(self):
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        current_q_list = []
        target_q_list = []

        for state, action, next_state, reward, done in zip(
            batch.state, batch.action, batch.next_state, batch.reward, batch.done
        ):
            state_t      = self.state_to_tensor(state)
            next_state_t = self.state_to_tensor(next_state)

            if done:
                target = torch.FloatTensor([reward]).to(self.device)
            else:
                with torch.no_grad():
                    target = torch.FloatTensor(
                        [reward + GAMMA * self.target_net(next_state_t).max().item()]
                    ).to(self.device)

            current_q = self.policy_net(state_t)
            current_q_list.append(current_q)

            target_q = self.target_net(state_t).clone().detach()
            target_q[action] = target
            target_q_list.append(target_q)

        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1

        # Decay epsilon
        self.epsilon = max(EPSILON_END, self.epsilon - EPSILON_DECAY)

        # Sync target network every N steps
        if self.step_count % TARGET_SYNC == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
