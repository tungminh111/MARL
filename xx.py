import marl
from marl.coma import COMA
from marl import MARL
from marl.agent.coma_agent import COMAAgent
from marl.exploration import EpsGreedy

from soccer import DiscreteSoccerEnv
# Environment available here "https://github.com/blavad/soccer"
env = DiscreteSoccerEnv(nb_pl_team1=1, nb_pl_team2=1)

obs_s = env.observation_space
act_s = env.action_space
print(":::ACS", act_s)

# Custom exploration process
expl = EpsGreedy(eps_deb=1.,eps_fin=.3)

# Create two minimax-Q agents
agent1 = COMAAgent(1, act_s, agent_quan=2, lr_critic=0.00001, exploration=expl, name="SoccerJ1")
agent2 = COMAAgent(1, act_s, agent_quan=2, lr_critic=0.00001, exploration=expl, name="SoccerJ2")

# Create the trainable multi-agent system
mas = COMA(agents_list=[agent1, agent2])

# Assign MAS to each agent
agent1.set_mas(mas)
agent2.set_mas(mas)

# Train the agent for 100 000 timesteps
mas.learn(env, ep_iter=100, batch_size=1)

# Test the agents for 10 episodes
mas.test(env, nb_episodes=10, time_laps=0.5)

