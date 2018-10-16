



![algorithms](https://github.com/NervanaSystems/coach/blob/master/img/algorithms.png)

-------------
- Temporal difference(TD) learning (1988)
- Q‐learning (1998)
- BayesRL (2002)
- RMAX (2002)
- CBPI (2002)
- PEGASUS (2002)
- Least‐Squares Policy Iteration (2003)
- Fitted Q‐Iteration (2005)
- GTD (2009)
- UCRL (2010)
- REPS (2010)
- DQN (2014) - DeepMind

--------------

   - Value Optimization
       - [QR-DQN]
       - [DQN] - [[Slides](https://drive.google.com/file/d/0BxXI_RttTZAhVUhpbDhiSUFFNjg/view)]
       - [Bootstrapped DQN]
       - [DDQN]
       - [NEC]
       - [MMC]
       - [N-step Q Learning]
       - [PAL]
       - [Categorical DQN]
       - [NAF]
   - Policy Optimization
       - [Policy Gradient]
       - [Actor Critic]
         - [DDPG]
           - [HAC DDPG]
           - [DDPG with HER]
         - [Clipped PPO]
         - [PPO]
   - [DFP]
   - Imitation
       - [Behavioural cloning]
       - [Inverse Reinforcement Learning]
       - [Generative Adversarial Imitation Learning]
-----------

##### Value Optimization Agents
* [Deep Q Network (DQN)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)  ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/dqn_agent.py))
* [Double Deep Q Network (DDQN)](https://arxiv.org/pdf/1509.06461.pdf)  ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/ddqn_agent.py))
* [Dueling Q Network](https://arxiv.org/abs/1511.06581)
* [Mixed Monte Carlo (MMC)](https://arxiv.org/abs/1703.01310)  ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/mmc_agent.py))
* [Persistent Advantage Learning (PAL)](https://arxiv.org/abs/1512.04860)  ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/pal_agent.py))
* [Categorical Deep Q Network (C51)](https://arxiv.org/abs/1707.06887)  ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/categorical_dqn_agent.py))
* [Quantile Regression Deep Q Network (QR-DQN)](https://arxiv.org/pdf/1710.10044v1.pdf)  ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/qr_dqn_agent.py))
* [N-Step Q Learning](https://arxiv.org/abs/1602.01783) | **Distributed**  ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/n_step_q_agent.py))
* [Neural Episodic Control (NEC)](https://arxiv.org/abs/1703.01988)  ([code](rl_coach/agents/nec_agent.py))
* [Normalized Advantage Functions (NAF)](https://arxiv.org/abs/1603.00748.pdf) | **Distributed**  ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/naf_agent.py))


##### Policy Optimization Agents
* [Policy Gradients (PG)](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf) | **Distributed**  ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/policy_gradients_agent.py))
* [Asynchronous Advantage Actor-Critic (A3C)](https://arxiv.org/abs/1602.01783) | **Distributed**  ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/actor_critic_agent.py))
* [Deep Deterministic Policy Gradients (DDPG)](https://arxiv.org/abs/1509.02971) | **Distributed**  ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/ddpg_agent.py))
* [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf)  ([code](rl_coach/agents/ppo_agent.py))
* [Clipped Proximal Policy Optimization (CPPO)](https://arxiv.org/pdf/1707.06347.pdf) | **Distributed**  ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/clipped_ppo_agent.py))
* [Generalized Advantage Estimation (GAE)](https://arxiv.org/abs/1506.02438) ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/actor_critic_agent.py#L86))

##### General Agents
* [Direct Future Prediction (DFP)](https://arxiv.org/abs/1611.01779) | **Distributed**  ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/dfp_agent.py))

##### Imitation Learning Agents
* Behavioral Cloning (BC)  ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/bc_agent.py))

##### Hierarchical Reinforcement Learning Agents
* [Hierarchical Actor Critic (HAC)](https://arxiv.org/abs/1712.00948.pdf) ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/ddpg_hac_agent.py))

##### Memory Types
* [Hindsight Experience Replay (HER)](https://arxiv.org/abs/1707.01495.pdf) ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/memories/episodic/episodic_hindsight_experience_replay.py))
* [Prioritized Experience Replay (PER)](https://arxiv.org/abs/1511.05952) ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/memories/non_episodic/prioritized_experience_replay.py))

##### Exploration Techniques
* E-Greedy ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/exploration_policies/e_greedy.py))
* Boltzmann ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/exploration_policies/boltzmann.py))
* Ornstein–Uhlenbeck process ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/exploration_policies/ou_process.py))
* Normal Noise ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/exploration_policies/additive_noise.py))
* Truncated Normal Noise ([code](rl_coach/exploration_policies/truncated_normal.py))
* [Bootstrapped Deep Q Network](https://arxiv.org/abs/1602.04621)  ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/agents/bootstrapped_dqn_agent.py))
* [UCB Exploration via Q-Ensembles (UCB)](https://arxiv.org/abs/1706.01502) ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/exploration_policies/ucb.py))
* [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295) ([code](https://github.com/NervanaSystems/coach/blob/master/rl_coach/exploration_policies/parameter_noise.py))

------------
![awesome](https://raw.githubusercontent.com/tigerneil/awesome-deep-rl/master/images/awesome-drl.png)
---------
![landscape](https://raw.githubusercontent.com/tangzhenyu/Reinforcement-Learning-in-Robotics/master/images/landscape.jpeg)

-------------------

Misc

- [reference implementations](https://github.com/gopala-kr/reinforce-tf/blob/master/ref-implementations.md)
- [review papers](https://github.com/gopala-kr/reinforce-tf/blob/master/review-papers.md)
- [RL platforms](https://github.com/gopala-kr/DRL-Agents/blob/master/platforms.md)

-------------

- [Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures)
- [UC Berkeley: Deep Reinforcement Learning](http://rail.eecs.berkeley.edu/deeprlcourse/)
- [MIT 6.S094: Deep Learning for Self-Driving Cars](https://selfdrivingcars.mit.edu/)
- [Deep Reinforcement Learning and Control 
Spring 2017, CMU 10703](https://katefvision.github.io/#readings)
- [Sutton & Barto's: Reinforcement Learning: An Introduction](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
- [Algorithms for Reinforcement Learning](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf)
- [reinforcejs](https://cs.stanford.edu/people/karpathy/reinforcejs/index.html)

----------------------------------------

_**Author**_

Gopala KR / @gopala-kr

----------------------------------------
