import torch
import torch.nn as nn
from collections import deque
import random
from snake_pygame import SnakeGame


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(17, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 3)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

    
class ReplayMemory:
    def __init__(self, capacity):
        self.transition = deque(maxlen=capacity)
    
    def remember(self, state, action, reward, next_state, done):
        self.transition.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        k = min(batch_size, len(self.transition))
        return random.sample(self.transition, k)
    
class Agent:
    def __init__(self):
        self.n_games = 0
        self.model = Net()
        self.memory = ReplayMemory(100_000)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.target_model = Net()
        self.target_model.load_state_dict(self.model.state_dict())
        
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def get_action(self, state):
        epsilon = max(0, 300 - self.n_games)
        
        nombre = random.randint(0, 200)
        if nombre < epsilon:
            action = random.randint(0, 2)
            return action
    
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            q_value = self.model(state_tensor)
            resultat = torch.argmax(q_value).item()
            return resultat
    
    def train_step(self, state, action, reward, next_state, done):
            state_tensor = torch.tensor(state, dtype=torch.float)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float)
            pred = self.model(state_tensor)
            action_tensor = torch.tensor(action, dtype=torch.float)
            reward_tensor = torch.tensor(reward, dtype=torch.float)
            done_tensor = torch.tensor(done, dtype=torch.float)
            
            with torch.no_grad():
                next_pred = self.model(next_state_tensor)
            
            if done:
                target_value = reward
            else:
                target_value = reward + 0.99 * torch.max(next_pred)
                
            target = pred.clone()
            target[action] = target_value
            
            criterion = nn.MSELoss()
            loss = criterion(pred, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
    def train_long_memory(self):
        batch = self.memory.sample(1000)
        states, actions, rewards, next_states, dones = zip(*batch)
        states_tensor = torch.tensor(states, dtype=torch.float)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float)

        
        preds = self.model(states_tensor)
        
        with torch.no_grad():
            nexts_pred = self.target_model(next_states_tensor)
            nexts_pred_model = self.model(next_states_tensor)
        
        targets = preds.clone()
        
        for i in range(len(batch)):
            if dones[i]:
                target_value = rewards[i]
            
            else:
                best_action = torch.argmax(nexts_pred_model[i])
                target_value = rewards[i] + 0.99 * nexts_pred[i][best_action]
            targets[i][actions[i]] = target_value
            
        criterions = nn.MSELoss()
        loss = criterions(preds, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()