# Grid settings
GRID_SIZE   = 4
NUM_STATES  = GRID_SIZE * GRID_SIZE  # 16 states
NUM_ACTIONS = 4                       # Up, Down, Left, Right

# Network
HIDDEN_NODES = 64

# Training
EPISODES     = 1000
BATCH_SIZE   = 32
MEMORY_SIZE  = 1000
TARGET_SYNC  = 10       # sync target network every N steps

# Hyperparameters
LEARNING_RATE  = 0.001
GAMMA          = 0.9    # discount factor
EPSILON_START  = 1.0    # start fully random
EPSILON_END    = 0.01   # minimum exploration
EPSILON_DECAY  = 1/EPISODES  # linear decay
