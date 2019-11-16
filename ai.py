#NOTICE: This project was built with the help of Udemy (https://www.udemy.com/course/artificial-intelligence-az/)
# Please check out the link above to view the full project. I have made a couple optimizations to the neural network
# to make it more efficient. Read the readme file in this repo for more information!

# AI (Brain) for self driving car

# Import the libraries needed

import numpy as np
import random
import os   #for loading/saving model
import torch   #pytorch for neural network
import torch.nn as nn  #tools for neural network
import torch.nn.functional as F
import torch.optim as optim #optimizer for gradient descent
from torch.autograd import Variable #tensor variable conversion

# Neural network architecture

#inherit tools for neural network
class Network(nn.Module):
    
    #Constructor
    #input_size = num neurons (5)
    def __init__(self, input_size, nb_action):
        #inherit from nn
        super(Network, self).__init__()
        #specify input layer
        self.input_size = input_size
        #output neurons
        self.nb_action = nb_action
        #full connection 1,connect input neurons to hidden layer, 30= num neurons in hidden layer
        self.fc1 = nn.Linear(input_size, 30)
        #full connection 2, connect hidden layer to output layer
        self.fc2 = nn.Linear(30, nb_action)
        
        
    #activates neurons by returning Q values  (left, right, or straight)  
    def forward(self, state):
        #activate hidden neurons (x) via rectifier function
        x = F.relu(self.fc1(state))
        #output neurons
        q_values = self.fc2(x)
        return q_values
    
# Implementing Experience Replay
# Consider more than 1 state from the past
class ReplayMemory(object):
    
    #Constructor
    #capacity = max num transitions in mem of events
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    #add new states/event to memory    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            #overflow
           del self.memory[0]
           
    def sample(self, batch_size):
        #random samples from memory of fixed size
        #zip*:  reshapes list as states, actions, rewards
        samples = zip(*random.sample(self.memory, batch_size))
        #concatenate samples and convert tensor into Variable with tensor and gradient
        #apply this function for each sample
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
    
    
# Implementing Deep Q Learning
class Dqn():
    # Constructor
    # @params: input_size and nb_action are for NN
    # gamma is part of equation (delay coefficient)
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        #reward window mean of last 100 rewards
        self.reward_window = []
        #model is neural network
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        # for gradient descent from torch library, optimizes NN
            #learning rate = lr, decay, etc
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        #transition events. last state, last action, last reward
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0    #0, 1, or 2
        self.last_reward = 0    # -1< r < 1
        
        
    #determines which action to play
    def select_action(self, state):
        #softmax- select best action to play but explore different actions
        #state is a tensor. Include graient in graph, improves performance
        probs = F.softmax(self.model(Variable(state, volatile = True)) * 100)     # t = 7 (higher T is more certain action) set T to 0 to remove brain
        #random draw
        action = probs.multinomial()
        return action.data[0, 0]
        
    #learn function (takes in a transition!) take from memory
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        #collect each action played/chosen
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]  #max of q values of next state
        #now get target, what we are trying to get
        target = self.gamma * next_outputs + batch_reward
        #compute loss, error of prediction (temporal difference)
        td_loss = F.smooth_l1_loss(outputs, target)    #loss function for deep Q learning
        #back propogate this back into the neural network with stochastic descent
        self.optimizer.zero_grad()  # reinstantiate at each loop
        td_loss.backward(retain_variables = True)   #backpropogation
        self.optimizer.step()   #updates the weights
        
        
    #sets new action of the brain
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)   #new state (a tensor) from signals received from car
        #append to memory
        self.memory.push((self.last_state, new_state, torch.LongTensor([int (self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)  #get the action based on the state
        #learn from the actions in the last 100 events
        if len(self.memory.memory) > 100:
            # we can learn from 100 transitions of memory
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
            
        #update existing variables
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward   # going onto the sand (bad) or not (good)
        self.reward_window.append(reward)   # add to reward window
        #cap it at 1000 size
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
            
        return action
    
    #average of rewards (score)
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.) # +1 to ensure != 0
    
    #save neural network and optimizer
    def save(self):
        # save neural network and optimizer (weights)
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                    }, 'last_brain.pth')
    
    #load last_brain file (neural network and optimizer)
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading model...")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])    #update neural network model and its weights
            self.optimizer.load_state_dict(checkpoint['optimizer'])    #update optimizer parameters
            print("done loading!")
        else:
            print("No checkpoint found (no last brain file found) ... ")
    
        
        
        
        
        
        
        
        
        
        
    
        
        
    