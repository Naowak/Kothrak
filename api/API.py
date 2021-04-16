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
    data = {'names': list({**agents, **new_agents}.keys())}
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
    state, _ = games[gid]._get_observation()
    state = torch.tensor(state, device=device).view(1, -1)

    action = agents[agent_name].play(state)
    _, _, done, infos = games[gid].step(action)

    if done:
        games.remove(gid)

    data = {'gid': gid, **infos}
    print(data)
    return cors(data)


@api.route('/human_play', methods=['GET'])
def human_play():
    gid, move, build = retrieve_args(gid=int, move='cell', build='cell')
    action = transform_actions_into_number(move, build)
    _, _, done, infos = games[gid].step(action)

    if done:
        games.remove(gid)

    data = {'gid': gid, **infos}
    return cors(data)


@api.route('/train', methods=['GET'])
def train():
    nb_agents, grid_ray = retrieve_args(nb_agents=int, grid_ray=int)
    env = KothrakEnv(nb_agents, grid_ray)
    session = Training()
    session.trainer = Trainer(env)
    tid = trainings.add(session)

    def task(trainer, tid):
        trainer.run()
        trainings.remove(tid)

    session.thread = Thread(target=task, args=(session.trainer, tid))
    session.thread.start()

    data = {'tid': tid, 'status': 'start'}
    return cors(data)


@api.route('/watch_training', methods=['GET'])
def watch_training():
    [tid] = retrieve_args(tid=int)
    while len(trainings[tid].trainer.replay) == 0:
        sleep(0.1)
    game_history = trainings[tid].trainer.last_replay()
    data = {'tid': tid, 'status': 'watch', 'history': game_history}
    return cors(data)



if __name__ == "__main__":
    api.run(debug=True)

    # Error managing :
    # Games and trainings disapear, it should return a message on the page instead
    # of an error 500
    # On the godot part : 
    # - not ask to replay if training is over
    # - make him play against an ai correctly by choosing an opponent in the list
    # then :
    # - pass to multi task (12 actions 2*6, sum of loss)
    # - a character loose if no play possible (empty list)
    # - blender work (color tower), create character
