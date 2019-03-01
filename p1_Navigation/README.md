## Set up
```python
# I use conda to manage virtual env
conda create --name drlnd python=3.6
source activate drlnd
python -m ipykernel install --user --name drlnd --display-name "drlnd"
git clone https://github.com/kaixinbaba/uda-drl.git
cd uda-drl
pip install -r requirements.txt
jupyter notebook
```
# Navigation
I tried a combination of various parameters to get the hyper parameters listed below. 
I can get an average of 13+ rewards after training for 1800 episodes.
I just use the origin DQN algorithm with **fix target** and **replay buffer**
## Network structure
Because of the use of DQN, the entire algorithm has two identical neural networks, 
one is eval_net and the other is target_net，
Since this environment is much more complicated than the traditional gym, 
So each neural network has two hidden layers of **128** and **64** respectively.
The input is the state of the environment, and the output is the expected reward of the action vector.
## Hyper parameters
- total_episode 1800
- learning rate 0.001
- replay_buffer_size 3000
- two hidden layers 128 and 64
- mini_batch_size 32
- epsilon 1.0
- epsilon_min 0.001
- epsilon_decay 0.99
- discount_gamma 0.99
- replace_iter 10
- very_small_number_e(0.01, Prevent division by 0)
- alpha 0.6
- beta 0.4
## Project structure
- Report.ipynb
- agents.py (RL agent by dueling DQN, prioritize)
- models.py (Just neural networks implement by pytorch)
- compat.py (compat gpu and cpu, simple module)
- memory.py (The replay buffer module implement by sumtree)
- model.pt (The agent train result saved model's file)
Then open the Report.ipynb to see the report.
