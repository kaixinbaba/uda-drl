import gym
import matplotlib.pyplot as plt

from agents import DDPG

if __name__ == '__main__':
    env = gym.make('Pendulum-v0').unwrapped
    env.seed(7)
    n_s = env.observation_space.shape[0]
    n_a = env.action_space.shape[0]
    a_bound = env.action_space.high[0]
    ddpg = DDPG(n_s,
                n_a,
                a_bound,
                gamma=0.9,
                memory_size=2000,
                tau=0.01,
                lr_a=0.001,
                lr_c=0.002,
                batch_size=32,
                var=3,
                var_decay=0.9995)
    rewards = []
    for e in range(200):
        print(e)
        s = env.reset()
        total_reward = 0
        for step in range(200):
            env.render()
            a = ddpg.choose_action(s)
            s_, r, done, _ = env.step(a)
            r /= 10
            total_reward += r
            ddpg.step(s, a, r, s_, done)
            if done:
                rewards.append(total_reward)
                break
            s = s_
    # e-r
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.title('result')
    plt.plot(range(len(rewards)), rewards)
    plt.show()
