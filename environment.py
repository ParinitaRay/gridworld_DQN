import random


class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.start = (0, 0)
        self.goal = (3, 3)
        self.dynamic_obstacles = 2   # number of attacks
        self.obstacles = []
        self.reset()
        self.generate_obstacles()

    def reset(self):
        self.agent_pos = self.start
        return self.state_index(self.agent_pos)

    def state_index(self, position):
        return position[0] * self.size + position[1]

    def generate_obstacles(self):
        """Randomly generate dynamic attack locations"""
        self.obstacles = []
        while len(self.obstacles) < self.dynamic_obstacles:
            r = random.randint(0, self.size - 1)
            c = random.randint(0, self.size - 1)
            pos = (r, c)
            if pos != self.start and pos != self.goal:
                if pos not in self.obstacles:
                    self.obstacles.append(pos)

    def step(self, action):
        row, col = self.agent_pos

        if action == 0:    # Up
            row -= 1
        elif action == 1:  # Down
            row += 1
        elif action == 2:  # Left
            col -= 1
        elif action == 3:  # Right
            col += 1

        # Boundary check
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            return self.state_index(self.agent_pos), -5, False

        new_pos = (row, col)

        # Attack detected
        if new_pos in self.obstacles:
            reward = -15
            done = False
        elif new_pos == self.goal:
            reward = 25
            done = True
        else:
            reward = -1
            done = False

        self.agent_pos = new_pos
        return self.state_index(new_pos), reward, done
