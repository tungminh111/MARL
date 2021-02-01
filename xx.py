import marl
from marl import MARL
from marl.agent import MinimaxQAgent
from marl.exploration import EpsGreedy

from soccer import DiscreteSoccerEnv
# Environment available here "https://github.com/blavad/soccer"
env = DiscreteSoccerEnv(nb_pl_team1=1, nb_pl_team2=1)

obs_s = env.observation_space
act_s = env.action_space

# Custom exploration process
expl = EpsGreedy(eps_deb=1.,eps_fin=.3)

# Create two minimax-Q agents
q_agent1 = MinimaxQAgent(obs_s, act_s, act_s, exploration=expl, gamma=0.9, lr=0.001, name="SoccerJ1")
q_agent2 = MinimaxQAgent(obs_s, act_s, act_s, exploration=expl, gamma=0.9, lr=0.001, name="SoccerJ2")

# Create the trainable multi-agent system
mas = MARL(agents_list=[q_agent1, q_agent2])

# Assign MAS to each agent
q_agent1.set_mas(mas)
q_agent2.set_mas(mas)

# Train the agent for 100 000 timesteps
mas.learn(env, nb_timesteps=1000000)

# Test the agents for 10 episodes
mas.test(env, nb_episodes=10, time_laps=0.5)

