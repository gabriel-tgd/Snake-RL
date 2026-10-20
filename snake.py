from mainsnake import Agent, ReplayMemory, Net
from snake_pygame import SnakeGame
import torch


checkpoint = torch.load(r"")

model = Net()
model.load_state_dict(checkpoint["model_state"])
model.eval()
print(f"Score du meilleur model : {checkpoint['score']}")

env = SnakeGame(render=True)
state = env.reset()
done = False
n_games = 1

while True:
    state_tensor = torch.tensor(state, dtype=torch.float)
    q_value = model(state_tensor)
    
    action = torch.argmax(q_value).item()
    state, reward, done = env.step(action)
    
    if done:
        n_games += 1
        print(f"Partie {n_games}, score: {env.score}")
        state = env.reset()