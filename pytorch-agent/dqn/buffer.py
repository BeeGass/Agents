from random import sample

class Replay_Buffer():
    def __init__(self, capacity=35000, mini_batch_size=64):
        self.rb = []
        self.capacity = capacity
        self.mini_batch_size = mini_batch_size

    def get_rb_batch(self):
        sample = sample(self.rb, self.mini_batch_size)
        states, actions, next_states, rewards, done = zip(*sample[: self.mini_batch_size])
        return states, actions, next_states, rewards, done
    
    def add_to_rb(self, new_transition):
        if len(self.rb) >= self.capacity:
            del self.rb[0] 
        self.rb.append(new_transition)
