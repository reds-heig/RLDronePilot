import multiprocessing as mp
import numpy as np


class SharedReplayBuffer:
    def __init__(self, input_dims, n_actions, max_size=10_000):
        self.mem_cntr = mp.Value('i', 0)
        self.lock = mp.Lock()
        self.condition = mp.Condition(self.lock)

        self.mem_size = max_size // max(np.prod(input_dims), n_actions)
        self.input_dims = np.prod(input_dims)
        self.n_actions = n_actions
        
        self.state_memory = mp.Array('d', np.zeros(self.mem_size * self.input_dims))
        self.new_state_memory = mp.Array('d', np.zeros(self.mem_size * self.input_dims))
        self.action_memory = mp.Array('d', np.zeros(self.mem_size * n_actions))
        self.reward_memory = mp.Array('d', np.zeros(self.mem_size))
        self.terminal_memory = mp.Array('d', np.zeros(self.mem_size))


    
    def store_transition(self, state, action, reward, state_, done):
        with self.lock:
            index = self.mem_cntr.value % self.mem_size
            self.mem_cntr.value += 1

            self.state_memory[index*self.input_dims : index*self.input_dims+self.input_dims] = state
            self.new_state_memory[index*self.input_dims : index*self.input_dims+self.input_dims] = state_
            self.action_memory[index*self.n_actions : index*self.n_actions+self.n_actions] = action
            self.reward_memory[index] = reward
            self.terminal_memory[index] = 1 - done
            
            self.condition.notify() # notify the LearningProcess a new transition has been pushed to the buffer


    def wait_for_transition(self):
        with self.condition:
            self.condition.wait() # wait for a new transition to be pushed to the buffer

    
    def sample_buffer(self, batch_size):
        with self.lock:
            max_mem = min(self.mem_cntr.value, self.mem_size)
            batch = np.random.choice(max_mem, batch_size)

            states = np.frombuffer(self.state_memory.get_obj()).reshape((-1, self.input_dims))[batch]
            actions = np.frombuffer(self.action_memory.get_obj()).reshape((-1, self.n_actions))[batch]
            rewards = np.frombuffer(self.reward_memory.get_obj())[batch]
            states_ = np.frombuffer(self.new_state_memory.get_obj()).reshape((-1, self.input_dims))[batch]
            terminal = np.frombuffer(self.terminal_memory.get_obj())[batch]

        return states, actions, rewards, states_, terminal


    def is_available(self, batch_size):
        with self.lock:
            return self.mem_cntr.value >= batch_size