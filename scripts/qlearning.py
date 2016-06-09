import numpy as np
import random
from copy import deepcopy

class Q_node():
    def __init__(self, learning_rate):
        self.values = {'N': 0, 'S': 0, 'E': 0, 'W': 0}
        self.counts = {'N': 1, 'S': 1, 'E': 1, 'W': 1}
        self.learning_rate = learning_rate

    def update_value(self, action, utility):
        new_value = self.learning_rate * utility
        self.values[action] = (1 - self.learning_rate) * self.values[action] + new_value
        self.counts[action] = self.counts[action] + 1

    def get_max_action_value(self):
        return self._get_max_action_value_pair()[1]

    def get_max_action(self):
        return self._get_max_action_value_pair()[0]

    def get_exploration_reward(self):
        max_value = None
        max_action = None
        for action in ['N', 'S', 'E', 'W']:
            if self.values[action] + (10/self.counts[action]) >= max_value:
                max_value = self.values[action] + (10/self.counts[action])
                max_action = action

        return (max_action, max_value)

    def _get_max_action_value_pair(self):
        max_value = None
        max_action = None
        for action in ['N', 'S', 'E', 'W']:
            if self.values[action] >= max_value:
                max_value = self.values[action]
                max_action = action

        return (max_action, max_value)

def out_of_bounds(height, width, pos):
    return pos[0] < 0 or pos[0] >= height or pos[1] < 0 or pos[1] >= width


def probabilistic_move(config, action):
    # determine which direction actually happens
    d = zip(deepcopy(config['direction_probs']), config['directions'][action])
    random.shuffle(d)
    ranges = [0] * 4
    for i in range(0, len(d)):
        for j in range(i, len(d)):
            ranges[j] += d[i][0]

    select = random.random()
    for i in range(len(ranges)):
        if select <= ranges[i]:
            return d[i][1]

def _find_tel_spot(pos, config):
    height, width = config['map_size']
    i,j = config['telmap'][pos]
    for di,dj in config['moves']:
        new_pos  = (i + di, j + dj)
        if out_of_bounds(height, width, new_pos) or new_pos in config['walls'] or new_pos in config['tel_dests'] or new_pos in config['pits']:
            continue
        return new_pos 

    

def move_random(config, nodes, current_pos):
    height, width = config['map_size']
    action = nodes[current_pos[0]][current_pos[1]].get_exploration_reward()[0]
    actual_direction = probabilistic_move(config, action)
    if actual_direction == None:
        print "I messed up"
        return

    new_i = current_pos[0] + actual_direction[0]
    new_j = current_pos[1] + actual_direction[1]
    new_pos = (new_i, new_j)
    reward = config['reward_for_each_step']
    if out_of_bounds(height, width, new_pos) or new_pos in config['walls'] or new_pos in config['tel_dests']:
        return (current_pos, action, config['reward_for_hitting_wall'])
    elif new_pos in config['pits']:
        return (new_pos, action, reward + config['reward_for_falling_in_pit'])
    elif new_pos == config['goal']:
        return (new_pos, action, reward + config['reward_for_reaching_goal'])
    elif new_pos in config['telmap']:
        return (_find_tel_spot(new_pos, config), action, reward + config['reward_for_teleporting'])
    else:
        return (new_pos, action, reward)

def get_policy(config, nodes, i, j):
    if (i,j) in config['pits']:
        return 'PIT'
    elif (i,j) in config['walls']:
        return 'WALL'
    elif (i,j) == config['goal']:
        return 'GOAL'
    elif (i,j) in config['tel_sources']:
        return 'T'
    elif (i,j) in config['tel_dests']:
        return 'D'
    else:
        return nodes[i][j].get_max_action()

    
def qlearning(config):
    # set up empty Q nodes
    height,width = config['map_size']   
    random.seed(config['seed'])
    nodes = []
    for i in range(height):
        nodes.append([])
        for j in range(width):
            nodes[i].append(Q_node(config['learning_rate']))
    nodes = np.array(nodes)

    # some setup for ease of use
    config['goal'] = tuple(config['goal'])
    config['pits'] = set(map(tuple, config['pits']))
    config['walls'] = set(map(tuple, config['walls']))
    config['moves'] = map(tuple, config['move_list'])
    config['actions'] = {(0,1) : 'E', (0,-1) : 'W', (1,0) : 'S', (-1,0) : 'N'}
    config['tel_sources'] = map(tuple, config['tel_sources'])
    config['tel_dests'] = map(tuple, config['tel_dests'])
    config['telmap'] = dict(zip(config['tel_sources'], config['tel_dests']))
    # directions are in order: forward/backward/left/right
    config['directions'] = { 'E' : [(0,1), (0,-1), (-1,0), (1,0)],
                             'W' : [(0,-1), (0,1), (1,0), (-1,0)],
                             'S' : [(1,0), (-1,0), (0,1), (0,-1)],
                             'N' : [(-1,0), (1,0), (0,-1), (0,1)] }
    config['direction_probs'] = [ config['prob_move_forward'],
                                  config['prob_move_backward'],
                                  config['prob_move_left'],
                                  config['prob_move_right'] ]
    
    # perform learning runs
    run = 0
    index = 0
    policies = []
    while run < config['max_runs']:
        current_pos = tuple(config['start'])
        walk = 0
        # stop when robot lands in pit or goal or up to 100 moves
        while current_pos not in config['pits'] and current_pos != config['goal'] and walk < 200:
            new_pos, action, reward = move_random(config, nodes, current_pos)
            future_reward = nodes[new_pos[0]][new_pos[1]].get_exploration_reward()[1]
            utility = reward + (config['discount_factor'] * future_reward)
            nodes[current_pos[0]][current_pos[1]].update_value(action, utility)
            current_pos = new_pos
            walk += 1
            index += 1

        policies.append([])
        for i in range(height):
            for j in range(width):
                if (i,j) == current_pos:
                    policies[run].append('R')
                else:
                    policies[run].append(get_policy(config, nodes, i, j))


        run += 1

    return policies
