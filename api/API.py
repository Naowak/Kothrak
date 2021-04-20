from flask import Flask
import torch
from threading import Thread
from time import sleep

from envs.KothrakEnv import KothrakEnv, transform_actions_into_number
from api.Utils import Manager, Training, retrieve_args, cors, \
    load_all_agents
from dqn.Trainer import Trainer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
api = Flask(__name__)
games = Manager()
trainings = Manager()
agents = load_all_agents()


@api.route('/agents_infos', methods=['GET'])
def agents_infos():
    global agents
    new_agents = load_all_agents(agents)
    agents.update(new_agents)
    data = {'status': 'agents_infos', 'names': list(agents.keys())}
    return cors(data)


@api.route('/new_game', methods=['GET'])
def new_game():
    nb_players, grid_ray = retrieve_args(nb_players=int, grid_ray=int)
    env = KothrakEnv(nb_players, grid_ray)
    gid = games.add(env)

    _, infos = env.reset()
    data = {'gid': gid, **infos}
    return cors(data)


@api.route('/agent_play', methods=['GET'])
def agent_play():
    gid, agent_name = retrieve_args(gid=int, agent_name=str)
    game = games[gid]
    if game is None:
        return cors({'Error': f'Game {gid} does not exist.'})

    state, _ = game._get_observation()
    state = torch.tensor(state, device=device).view(1, -1)

    action = agents[agent_name].play(state)
    _, _, done, infos = game.step(action)

    if done:
        games.remove(gid)

    data = {'gid': gid, **infos}
    print(data)
    return cors(data)


@api.route('/human_play', methods=['GET'])
def human_play():
    gid, move, build = retrieve_args(gid=int, move='cell', build='cell')
    game = games[gid]
    if game is None:
        return cors({'status': 'GameIDError'})

    action = transform_actions_into_number(move, build)
    _, _, done, infos = game.step(action)

    if done:
        games.remove(gid)

    data = {'gid': gid, **infos}
    return cors(data)


@api.route('/train', methods=['GET'])
def train():
    nb_agents, grid_ray, agent_names = retrieve_args(nb_agents=int, 
                                            grid_ray=int, agent_names=str)
    agent_names = agent_names.split(',')
    env = KothrakEnv(nb_agents, grid_ray)
    session = Training()
    session.trainer = Trainer(env, agent_names, nb_agents)
    tid = trainings.add(session)

    def task(trainer, tid, agents):
        trainer.run()
        trainings.remove(tid)

    session.thread = Thread(target=task, args=(session.trainer, tid, agents))
    session.thread.start()

    data = {'tid': tid, 'status': 'start'}
    return cors(data)


@api.route('/watch_training', methods=['GET'])
def watch_training():
    [tid] = retrieve_args(tid=int)
    training = trainings[tid]
    if training is None:
        return cors({'status': 'TrainingIDError'})

    while len(training.trainer.replay) == 0:
        sleep(0.1)
    game_history = training.trainer.last_replay()
    data = {'tid': tid, 'status': 'watch', 'history': game_history}
    return cors(data)



if __name__ == "__main__":
    api.run(debug=True)

    # - pass to multi task (12 actions 2*6, sum of loss)
    # - a character loose if no play possible (empty list)
    # - blender work (color tower), create character
    # - code rewiew : comments
