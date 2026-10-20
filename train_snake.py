from mainsnake import Agent, ReplayMemory
from snake_pygame import SnakeGame
import torch


env = SnakeGame(render=False)
agent = Agent()
state = env.reset()
print("L'entrainement commence ! ")

record = 0

print(len(state))
                                                                                                               
fichier = ""                                                                                                                                                                                                                        

with open(f"logging_{fichier}.txt", "a") as a:

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        agent.train_step(state, action, reward, next_state, done)
        agent.memory.remember(state, action, reward, next_state, done)
                
        if done:
            print("Partie", agent.n_games, "score", env.score)
            a.write(f"parti : {agent.n_games}, score : {env.score}\n")
            agent.train_long_memory()
            agent.n_games += 1
            if agent.n_games % 100 == 0:
                agent.update_target()
            if env.score > record:
                record = env.score
                torch.save({
                    "model_state": agent.model.state_dict(),
                    "score": record,
                    }, f"{fichier}.pth")
            print(f"record : {record}")
            state = env.reset()
    
        else:   
            state = next_state