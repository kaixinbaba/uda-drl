# Navigation
I use the DQN algorithm and an improved version(prioritize replay buffer, dueling net work) that uses it
to finish the project.
## Network structure
Because of the use of DQN, the entire algorithm has two identical neural networks, 
one is eval_net and the other is target_net，
Since this environment is much more complicated than the traditional gym, 
So each neural network has two hidden layers of **128** and **64** respectively.
The input is the state of the environment, and the output is the expected reward of the action vector.
## Hyper parameters
- total_episode 2000
- learning rate 0.001
- replay_buffer_size 3000
- mini_batch_size 64
- epsilon 1.0
- epsilon_min 0.01
- epsilon_decay 0.99
- discount_gamma 0.99
- tau 0.01 (Use soft update the target net'param from eval)
- very_small_number_e(0.01, Prevent division by 0)
- alpha 0.6
- beta 0.4
## Project structure
- Report.ipynb (Main report and can run last cell to test the agent, use 'model.pt'')
- agents.py (RL agent by dueling DQN, prioritize)
- models.py (Just neural networks implement by pytorch)
- compat.py (compat gpu and cpu, simple module)
- memory.py (The replay buffer module implement by sumtree)
- model.pt (The agent train result saved model's file)
## Set up
```python
pip install -r requirements.txt
jupyter notebook
```
Then open the Report.ipynb to see the report.
