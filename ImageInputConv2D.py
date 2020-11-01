from __future__ import division

#import imageio
import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
import matplotlib.pyplot as plt
import threading
import time
import scipy.signal as signal
import GroupLock
import multiprocessing
import scipy
#get_ipython().run_line_magic('matplotlib', 'inline')

from tensorflow.python.client import device_lib
dev_list = device_lib.list_local_devices()
print(dev_list)
# assert len(dev_list) > 1


from gym import spaces
from collections import OrderedDict
from threading import Lock
import sys
import pyglet
from main_Vision import convertEBtoSB, Window
import pickle, os
import time
#from skimage.transform import resize

WINDOWS = []



'''
3D Grid Environment
    Observation: (position maps of current agent, other agents, blocks, sources, and plan)
        Position:   X, Y, Z  (+Y = up)
        View:       A box centered around the agent (limited view)
            block = -1
            block spawner = -2
            air = 0
            agent = 1 (agent_id in id_visible mode, agent_id is a positive integer)
            out of world range = -3

    Action space: (Tuple)
        agent_id: positive integer
        action: {0:NOP, 1:MOVE_NORTH, 2:MOVE_EAST, 3:MOVE_SOUTH, 4:MOVE_WEST, 5:PICK_NORTH,
        6:PICK_EAST, 7:PICK_SOUTH, 8:PICK_WEST, 9:PLACE_NORTH, 10:PLACE_EAST, 11:PLACE_SOUTH, 12:PLACE_WEST}
        NORTH: +Y, EAST: +X

    Reward: ACTION_COST for each action, REWARD for each block correctly placed
'''

'''
Rules (specifics):
A robot cannot walk on a block currently carried by an other robot.
A robot cannot place a block on an other robot or on a source.
A robot cannot walk over an other robot or a source.
A robot cannot be on the highest level of the simulation (world_shape[1]-1).
'''

PLAN_MAPS = [
## Simple 6x6 castle (1 block high) with 4 towers (3 blocks high)
    [[2, 0, 2], [2, 1, 2], [2, 2, 2], [2, 0, 3], [2, 0, 4], [2, 0, 5], [2, 0, 6], [7, 0, 2], [7, 1, 2], [7, 2, 2], [3, 0, 2], [3, 0, 7], [4, 0, 2], [4, 0, 7], [5, 0, 2], [5, 0, 7], [6, 0, 2], [6, 0, 7], [2, 0, 7], [2, 1, 7], [2, 2, 7], [7, 0, 3], [7, 0, 4], [7, 0, 5], [7, 0, 6], [7, 0, 7], [7, 1, 7], [7, 2, 7]],
## 4 1x3 towers only
    [[2, 0, 2], [2, 1, 2], [2, 2, 2], [7, 0, 2], [7, 1, 2], [7, 2, 2], [2, 0, 7], [2, 1, 7], [2, 2, 7], [7, 0, 7], [7, 1, 7], [7, 2, 7]],
## Pyramid centered and 2x2x3 high at the middle
    [[2, 0, 2], [2, 0, 3], [2, 0, 4], [2, 0, 5], [2, 0, 6], [2, 0, 7], [3, 0, 2], [3, 0, 3], [3, 0, 4], [3, 0, 5], [3, 0, 6], [3, 0, 7], [4, 0, 2], [4, 0, 3], [4, 0, 4], [4, 0, 5], [4, 0, 6], [4, 0, 7], [5, 0, 2], [5, 0, 3], [5, 0, 4], [5, 0, 5], [5, 0, 6], [5, 0, 7], [6, 0, 2], [6, 0, 3], [6, 0, 4], [6, 0, 5], [6, 0, 6], [6, 0, 7], [7, 0, 2], [7, 0, 3], [7, 0, 4], [7, 0, 5], [7, 0, 6], [7, 0, 7], [3, 1, 3], [3, 1, 4], [3, 1, 5], [3, 1, 6], [4, 1, 3], [4, 1, 4], [4, 1, 5], [4, 1, 6], [5, 1, 3], [5, 1, 4], [5, 1, 5], [5, 1, 6], [6, 1, 3], [6, 1, 4], [6, 1, 5], [6, 1, 6], [4, 2, 4], [4, 2, 5], [5, 2, 4], [5, 2, 5]],
## 1 big center cube (3x3x3)
    [[3, 0, 3], [3, 1, 3], [3, 2, 3], [3, 0, 4], [3, 1, 4], [3, 2, 4], [3, 0, 5], [3, 1, 5], [3, 2, 5], [4, 0, 3], [4, 1, 3], [4, 2, 3], [4, 0, 4], [4, 1, 4], [4, 2, 4], [4, 0, 5], [4, 1, 5], [4, 2, 5], [5, 0, 3], [5, 1, 3], [5, 2, 3], [5, 0, 4], [5, 1, 4], [5, 2, 4], [5, 0, 5], [5, 1, 5], [5, 2, 5]],
## 1 big cross wall (1x6x3 + 6x1x3)
    [[2, 0, 4], [2, 1, 4], [2, 2, 4], [3, 0, 4], [3, 1, 4], [3, 2, 4], [4, 0, 4], [4, 1, 4], [4, 2, 4], [5, 0, 4], [5, 1, 4], [5, 2, 4], [6, 0, 4], [6, 1, 4], [6, 2, 4], [7, 0, 4], [7, 1, 4], [7, 2, 4], [4, 0, 2], [4, 1, 2], [4, 2, 2], [4, 0, 3], [4, 1, 3], [4, 2, 3], [4, 0, 4], [4, 1, 4], [4, 2, 4], [4, 0, 5], [4, 1, 5], [4, 2, 5], [4, 0, 6], [4, 1, 6], [4, 2, 6], [4, 0, 7], [4, 1, 7], [4, 2, 7]],
## Center colulmn
    [[4, 0, 4], [4, 1, 4], [4, 2, 4], [4, 0, 5], [4, 1, 5], [4, 2, 5], [5, 0, 4], [5, 1, 4], [5, 2, 4], [5, 0, 5], [5, 1, 5], [5, 2, 5]]
]
SOURCES = [[0, 0, 0], [0, 0, 9], [9, 0, 0], [9, 0, 9]]

opposite_actions = {0: 0, 1: 3, 2: 4, 3: 1, 4: 2, 5: 9, 6: 10, 7: 11, 8: 12, 9: 5, 10: 6, 11: 7, 12: 8}
ACTION_COST, PLACE_REWARD = -0.02, +1.

BLOCK = np.array((210,105,30)) / 256.0
AIR = np.array((250,250,250)) / 256.0
PLAN_COLOR = np.array((250, 100, 100)) / 256.0
BLOCK_SPAWN = np.array((220,20,60)) / 256.0
AGENT = np.array((50,205,50)) / 256.0
OUT_BOUNDS = np.array((189,183,107)) / 256.0

class Grid3DState(object):
    '''
    3D Grid State.
    Implemented as a 3d numpy array.
        ground = -3
        block spawner = -2
        air = -1
        block = 0
        agent = positive integer (agent_id)
    '''
    def __init__(self, world0, num_agents=1):
        self.state = world0.copy()
        self.shape = np.array(world0.shape)
        self.num_agents = num_agents
        self.scanWorld()

    # Scan self.state for agents and load them into database
    def scanWorld(self):
        agents_list = []
        self.agents_pos = np.zeros((self.num_agents+1,3)) # x,y,z of each agent at start

        # list all agents
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    val = self.getBlock([i, j, k])
                    if val > 0:
                        assert val not in agents_list, 'ID conflict between agents'
                        assert type(val) is int or float, 'Non-integer agent ID'
                        val = int(val)
                        agents_list.append(val)
                        self.agents_pos[val] = [i,j,k]

        assert len(agents_list) == self.num_agents, 'Incorrect number of agents found in initial world'

    # Get value of block
    def getBlock(self, coord):
        # change coordinates to int
        coord = np.array(coord, dtype=int)

        if (coord < 0).any() or (coord >= self.shape).any():
            return -3
        return self.state[coord[0], coord[1], coord[2]]

    # Set block to input value
    def setBlock(self, coord, val):
        # change coordinates to int
        coord = np.array(coord, dtype=int)

        if (coord < 0).any() or (coord >= self.shape).any():
            return False
        self.state[coord[0], coord[1], coord[2]] = val
        return True

    # Swap two blocks
    def swap(self, coord1, coord2, agent_id):
        temp = self.getBlock(coord1)
        if temp == -2:
            self.setBlock(coord2, -1)
        else:
            if self.getBlock(coord2) == -2:
                self.setBlock(coord1, 0)
            else:
                self.setBlock(coord1, self.getBlock(coord2))
                self.setBlock(coord2, temp)

    # Get value of block
    def getPos(self, agent_id):
        # change coordinates to int
        coord = np.array(self.agents_pos[agent_id], dtype=int)

        return coord

    def setPos(self, new_pos, agent_id):
        self.agents_pos[agent_id] = new_pos
        npx, npy, npz = int(new_pos[0]), int(new_pos[1]), int(new_pos[2])
        assert self.state[npx, npy, npz] == agent_id, "Problem: agent {}'s position in agents_pos does not seem to match world.state ({})".format(agent_id, self.getBlock(new_pos))

    # Return predicted new state after action (Does not actually execute action, may be an invalid action)
    def act(self, action, agent_id):
        current_state = self.getPos(agent_id)
        new_state = current_state.copy()

        # Move
        if action in range(1,5):
            new_state[0:3] += self.heading2vec(action-1)

        return new_state

    # Get observation
    def getObservation(self, coord, ob_range):
        '''
        Observation: Box centered around agent position
            (returns -3 for blocks outside world boundaries)

        args:
            coord: Position of agent. Numpy array of length 3.
            ob_range: Vision range. Numpy array of length 3.

        note: observation.shape is (2*ob_range[0]+1, 2*ob_range[1]+1, 2*ob_range[2]+1)
        '''
        if (ob_range == [-1, -1, -1]).all(): # see EVERYTHING
            world_state = self.state
        else:
            ob = -3*np.ones([2*ob_range[0]+1, 2*ob_range[1]+1, 2*ob_range[2]+1])

            # change coordinates to int
            coord = np.array(coord, dtype=int)

            # two corners of view in world coordinate
            c0 = coord - ob_range
            c1 = coord + ob_range

            # clip according to world boundaries
            c0_c = np.clip(c0, [0,0,0], self.shape)
            c1_c = np.clip(c1, [0,0,0], self.shape)

            # two corners of view in observation coordinates
            ob_c0 = c0_c - coord + ob_range
            ob_c1 = c1_c - coord + ob_range

            # assign data from world to observation
            world_state = self.state[c0_c[0]:c1_c[0]+1, c0_c[1]:c1_c[1]+1, c0_c[2]:c1_c[2]+1]
            ob[ob_c0[0]:ob_c1[0]+1, ob_c0[1]:ob_c1[1]+1, ob_c0[2]:ob_c1[2]+1] = world_state

        return world_state

    # Compare with a plan to determine job completion
    def done(self, state_obj):
        blocks_state = np.asarray(np.clip(self.state, -1., 0.), dtype=int)
        blocks_plan  = np.asarray(np.clip( state_obj, -1., 0.), dtype=int)

        is_built = np.sum(blocks_state * blocks_plan) == -np.sum(blocks_plan) # All correct blocks are placed
        done = (blocks_state == blocks_plan).all()

        return done, is_built
    
    def countExtraBlocks(self, state_obj):
        blocks_state = np.asarray(np.clip(self.state, -1., 0.), dtype=int)
        blocks_plan  = np.asarray(np.clip( state_obj, -1., 0.), dtype=int)

        return (np.sum(blocks_plan) - np.sum(blocks_state))

    # Transform heading to x, z
    def heading2vec(self, fac):
        dx = ((fac + 1) % 2)*(1 - fac)
        dy = 0
        dz = (fac % 2)*(2 - fac)
        return np.asarray([dx,dy,dz])


class MinecraftEnv(gym.Env):
    '''
    3D Grid Environment
        Observation: (OrderedDict)
            Position:   X, Y, Z  (+Y = up)
            Action heading:     {0:+Z, 1:+X, 2:-Z, 3:-X}
            View:       A box centered around the agent (limited view)
                block = -1
                air = 0
                agent = 1
                out of world range = -3
        Action space: (Tuple)
            agent_id: positive integer (always 1)
            action: {0:NOP, 1:MOVE_NORTH, 2:MOVE_EAST, 3:MOVE_SOUTH, 4:MOVE_WEST, 5:PICK_NORTH,
            6:PICK_EAST, 7:PICK_SOUTH, 8:PICK_WEST, 9:PLACE_NORTH, 10:PLACE_EAST, 11:PLACE_SOUTH, 12:PLACE_WEST}
            NORTH: +Y, EAST: +X
        Reward: -0.1 for each action, +5 for each block correctly placed
    '''
    metadata = {"render.modes": ["human", "ansi"]}

    # Initialize env
    def __init__(self, num_agents=1, observation_range=1, observation_mode='id_visible', world0=None, FULL_HELP = False, MAP_ID=1):
        """
        Args:
            observation_range: Integer for cube. List of length 3 for box.
            observation_mode: {'default', 'id_visible'}
        """
        # Parse input parameters and check if valid
        #   observation_range
        if type(observation_range) is int:
            ob_range = observation_range*np.ones(3, dtype=int)
        else:
            assert len(observation_range) == 3, 'Wrong number of dimensions for \'observation_range\''
            ob_range = np.array(observation_range)

        #   observation_mode
        assert observation_mode in ['default', 'id_visible'], 'Invalid \'observation_mode\''

        # Initialize member variables
        self.num_agents = num_agents
        #self.ob_shape = 2*ob_range + 1
        self.ob_range = ob_range
        self.ob_mode = observation_mode
        self.finished = False
        self.mutex = Lock()
        self.fresh = False

        self.FULL_HELP = FULL_HELP # Defines if we help agent identify its next goal
        self.map_id = MAP_ID-1
        self.RANDOMIZED_PLANS = (MAP_ID == 0) # Defines if we randomize the plans during training
        
        # Initialize data structures
        self.world_shape = (10,4,10)
        self._setObjective()
        if world0 is None:
            self._setInitial()
        else:
            self.state_init = world0

        # Check everything is alright
        assert self.state_init.shape == self.state_obj.shape, '\'state_init\' and \'state_obj\' dimensions do not match'
        self.world = Grid3DState(self.state_init, self.num_agents)

        self.action_space = spaces.Tuple([spaces.Discrete(self.num_agents), spaces.Discrete(13)])
        self.viewer = None

    # Define objective world here
    def _setObjective(self):
        '''
        Objective state of the world (3d numpy array)
            air = 0
            block = -1
        '''
        plan_map = np.zeros(self.world_shape)

        if self.RANDOMIZED_PLANS:
            p_plan = np.random.uniform(0.05, 0.5)

            while np.sum(plan_map[:,0,:]) == 0:
                randPlan = - np.random.binomial(1, p_plan, size=self.world_shape)
                plan_map[:,0,:] = randPlan[:,0,:]

            # source block (nothing around to allow pickup)
            for pos in SOURCES:
                plan_map[pos[0], pos[1], pos[2]] = -2
                if pos[0]-1 >= 0:
                    plan_map[pos[0]-1, pos[1], pos[2]] = 0
                if pos[0]+1 < self.world_shape[0]:
                    plan_map[pos[0]+1, pos[1], pos[2]] = 0
                if pos[2]-1 >= 0:
                    plan_map[pos[0], pos[1], pos[2]-1] = 0
                if pos[2]+1 < self.world_shape[2]:
                    plan_map[pos[0], pos[1], pos[2]+1] = 0

            # Other random blocks
            for j in range(1, self.world_shape[1]-1): # blocks cannot be placed at the highest level
                # Let's place blocks on level j on top of blocks on level j-1 only
                plan_map[:,j,:] = (plan_map[:,j-1,:] == -1).astype(int) * randPlan[:,j,:]
        else:
            # Place blocks on world plan
            for pos in PLAN_MAPS[self.map_id]:
                plan_map[pos[0], pos[1], pos[2]] = -1

            # source block (nothing around to allow pickup)
            for pos in SOURCES:
                plan_map[pos[0], pos[1], pos[2]] = -2
                if pos[0]-1 >= 0:
                    plan_map[pos[0]-1, pos[1], pos[2]] = 0
                if pos[0]+1 < self.world_shape[0]:
                    plan_map[pos[0]+1, pos[1], pos[2]] = 0
                if pos[2]-1 >= 0:
                    plan_map[pos[0], pos[1], pos[2]-1] = 0
                if pos[2]+1 < self.world_shape[2]:
                    plan_map[pos[0], pos[1], pos[2]+1] = 0

        self.state_obj = plan_map

    # Define initial agent distribution here
    def _setInitial(self, empty=False, full=False):
        '''
        Initial state of the world (3d numpy array)
            air = 0
            block = -1
            source = -2
            agent = agent_id (always 1)
        '''
        # Randomized world based on self.state_obj
        #p_sparse, p_plan = 0.1, 0.4
        if full:
            p_sparse, p_plan = np.random.uniform(0., 0.5), 1.
        else:
            p_sparse, p_plan = np.random.uniform(0., 0.3), np.random.uniform(0., 1.)
        randSparse = np.random.binomial(1, p_sparse, size=self.world_shape)
        randPlan = np.random.binomial(1, p_plan, size=self.world_shape)

        world = np.zeros(self.world_shape)
        if not empty:
            world[:,0,:] = self.state_obj[:,0,:] * randPlan[:,0,:] + (-1-self.state_obj[:,0,:]) * randSparse[:,0,:]

        # source block (nothing around to allow pickup)
        for pos in SOURCES:
            world[pos[0], pos[1], pos[2]] = -2
            if pos[0]-1 >= 0:
                world[pos[0]-1, pos[1], pos[2]] = 0
            if pos[0]+1 < self.world_shape[0]:
                world[pos[0]+1, pos[1], pos[2]] = 0
            if pos[2]-1 >= 0:
                world[pos[0], pos[1], pos[2]-1] = 0
            if pos[2]+1 < self.world_shape[2]:
                world[pos[0], pos[1], pos[2]+1] = 0

        # agents: Random initial position
        for i in range(self.num_agents):
            rx, ry, rz = np.random.randint(self.world_shape[0]), np.random.randint(2), np.random.randint(self.world_shape[2])
            while not (world[rx,ry,rz] == 0 and ((ry == 0) or (ry > 0 and world[rx,ry-1,rz] == -1))):
                rx, ry, rz = np.random.randint(self.world_shape[0]), np.random.randint(self.world_shape[1]), np.random.randint(self.world_shape[2])
            world[rx,ry,rz] = i+1

        if not empty:
            # Other random blocks
            for j in range(1, self.world_shape[1]-1): # blocks cannot be placed at the highest level
                # Where are agents on level j
                agentMap = (world[:,j,:] > 0).astype(int) * world[:,j,:]
                # We can place blocks either on agents, or on blocks that are not themselves on agents. Also, let's not place blocks were agents are...
                if j < 2:
                    prevMap = (1-np.clip(agentMap,0,1)) * np.clip((world[:,j-1,:] > 0).astype(int) + (world[:,j-1,:] == -1).astype(int), 0, 1)
                else:
                    prevMap = (1-np.clip(agentMap,0,1)) * np.clip((world[:,j-1,:] > 0).astype(int) + (world[:,j-1,:] == -1).astype(int) * (world[:,j-2,:] == -1).astype(int), 0, 1)
                # Let's place blocks on level j
                world[:,j,:] = self.state_obj[:,j,:] * prevMap * randPlan[:,j,:] + (-1-self.state_obj[:,j,:]) * prevMap * randSparse[:,j,:] + agentMap

        self.state_init = world

    # Returns an observation of an agent
    def _observe(self, agent_id,window,is_built,MA_ID=-1):
        
        has_block = False

        agent_pos = self.world.getPos(agent_id)

        px, py, pz = int(agent_pos[0]), int(agent_pos[1]), int(agent_pos[2])

        agent_position = [px, py, pz]
        sx, sy, sz = self.world_shape[0]//2, self.world_shape[1]//2, self.world_shape[2]//2
        px_world ,py_world , pz_world = px-sx, py-sy, pz-sz
        state = self.world.state.copy()
        

        poses = np.where(self.state_obj==-1)
        poses = np.array(poses)
        poses = poses.T
                            
                            
        for pos in poses:
            if state[pos[0], pos[1], pos[2]] ==0:
                state[pos[0], pos[1], pos[2]] = -4
            elif state[pos[0], pos[1], pos[2]] ==-1 and [pos[0], pos[1], pos[2]]!= [px, py+1, pz]:
                state[pos[0], pos[1], pos[2]] = -3


        state_buffer = convertEBtoSB(state)
        state_buffer_stacked = np.vstack((state_buffer,state_buffer))

        window.set_State(state_buffer_stacked)

        window.set_position(px_world ,py_world , pz_world)
        window.on_key_press(112,0)
        
        window.set_position(px_world ,py_world , pz_world)

        # captures the images 
        screenshots = window.take_pics()
        
        if py+1 != self.world_shape[1]:
            if state[px,py+1,pz] == -1:
                has_block = True
            else:
                has_block = False
        else:
            has_block = False
            
        s_sur = screenshots[0:4,:,:,:]
        s_top = screenshots[4,:,:,:]
        s_top = np.expand_dims(s_top, axis=0)
            
        return s_sur , s_top , agent_position

    # Resets environment
    def _reset(self, agent_id, empty=False, full=False):
        self.finished = False
        self.mutex.acquire()

        if not self.fresh:
            # Check everything is alright
            assert self.state_init.shape == self.state_obj.shape, '\'state_init\' and \'state_obj\' dimensions do not match'

            # Initialize data structures
            self._setObjective()
            self._setInitial(empty=empty, full=full)
            self.world = Grid3DState(self.state_init, self.num_agents)
            #self._initSpaces()

            self.fresh = True
            self.finalAgentID = 0

        _, is_built = self.world.done(self.state_obj)
        has_block = self.world.getBlock(self.world.getPos(agent_id) + np.array([0,1,0])) == -1
        
        self.mutex.release()
        
        return self._listNextValidActions(agent_id), has_block, is_built

    # Executes an action by an agent
    def _step(self, action_input):

        self.fresh = False

        # Check action input
        assert len(action_input) == 3, 'Action input should be a tuple with the form (agent_id, action)'
        assert action_input[1] in range(13), 'Invalid action'
        assert action_input[0] in range(1, self.num_agents+1)

        # Parse action input
        agent_id = action_input[0]
        action   = action_input[1]
        window = action_input[2]
        
        
        # Lock mutex (race conditions start here)
        self.mutex.acquire()
        initDone = self.finished

        # Get current agent state
        agent_pos = self.world.getPos(agent_id)

        # Get estimated new agent state
        new_agent_pos = self.world.act(action, agent_id)

        # Execute action & determine reward
        reward = ACTION_COST

        if action in range(1,5):     # Move
            validAction = False # Valid Movement ?

            # get coordinates and blocks near new position
            new_pos = new_agent_pos
            new_pos_upper = new_pos + np.array([0,1,0])
            new_pos_lower = new_pos + np.array([0,-1,0])
            new_pos_lower2 = new_pos + np.array([0,-2,0])

            block_newpos = self.world.getBlock(new_pos)
            block_upper = self.world.getBlock(new_pos_upper)
            block_lower = self.world.getBlock(new_pos_lower)
            block_lower2 = self.world.getBlock(new_pos_lower2)

            # execute movement if valid
            if block_newpos == 0:  # air in front?
                if block_lower == -1 or block_lower == -3:    # block or ground beneath?
                    dest = np.array(new_pos, dtype=int)
                    if (self.world.state[dest[0], 0:dest[1], dest[2]] == -1).all():
                        new_agent_pos = new_pos    # horizontal movement
                        validAction = True
                elif block_lower == 0 and block_lower2 in [-1, -3]:   # block or ground beneath?
                    dest = np.array(new_pos_lower, dtype=int)
                    if (self.world.state[dest[0], 0:dest[1], dest[2]] == -1).all():
                        new_agent_pos = new_pos_lower  # downstairs movement
                        validAction = True
            elif block_newpos == -1 and block_upper == 0:   # block in front and air above?
                dest = np.array(new_pos_upper, dtype=int)
                if (self.world.state[dest[0], 0:dest[1], dest[2]] == -1).all():
                    new_agent_pos = new_pos_upper    #upstairs movement
                    validAction = True

            # Prevent agents from accessing the highest level
            if new_agent_pos[1] == self.world_shape[1]-1:
                validAction = False

            if validAction:
                self.world.swap(agent_pos, new_agent_pos, agent_id)
                self.world.swap(agent_pos + np.array([0,1,0]), new_agent_pos + np.array([0,1,0]), agent_id)
                self.world.setPos(new_agent_pos, agent_id)

        elif action in range(5,13):       # Pick & Place
            # determine block movement
            top = agent_pos + np.array([0,1,0])
            front = agent_pos + self.world.heading2vec((action-1) % 4)

            if action < 9: # pick
                source = front
                dest = top
            else:
                source = top
                dest = front
            above_source = source + np.array([0,1,0])
            dest = np.array(dest, dtype=int)
            source = np.array(source, dtype=int)

            # execute
            if self.world.getBlock(source) in [-1, -2] and self.world.getBlock(dest) in [0, -2] and self.world.getBlock(above_source) in [0, -3] and (action < 9 or (action > 8 and (self.world.state[dest[0], 0:dest[1], dest[2]] == -1).all())) and not (action < 9 and source[1] == self.world_shape[1]-1):
                if self.world.getBlock(dest) == -2: # Blocks can be destroyed by placing them in a source. However, we
                    self.world.setBlock(source, 0)  # should not use swap in this case (or agents will pick up the source)

                    if self.FULL_HELP:
                        if np.sum(np.clip(self.world.state, -1, 0)) > np.sum(np.clip(self.state_obj, -1, 0)):
                            reward -= PLACE_REWARD
                        elif np.sum(np.clip(self.world.state, -1, 0)) < np.sum(np.clip(self.state_obj, -1, 0)):
                            reward += PLACE_REWARD
                elif self.world.getBlock(source) == -2: # Make a block appear above the agent
                    self.world.setBlock(dest, -1)

                    if self.FULL_HELP:
                        if np.sum(np.clip(self.world.state, -1, 0)) < np.sum(np.clip(self.state_obj, -1, 0)):
                            reward -= PLACE_REWARD
                        elif np.sum(np.clip(self.world.state, -1, 0)) > np.sum(np.clip(self.state_obj, -1, 0)):
                            reward += PLACE_REWARD
                else:
                    self.world.swap(source, dest, agent_id)

                    # place/pick incorrect block creates additional +/- rewards only once plan is completed, to encourage cleanup
                    _, complete = self.world.done(self.state_obj)

                    if action > 8 and self.state_obj[dest[0], dest[1], dest[2]] == -1: # Place correct block
                        reward += PLACE_REWARD * (dest[1]+1)**2
                    elif action < 9 and self.state_obj[source[0], source[1], source[2]] == -1: # Removing correct block
                        reward -= PLACE_REWARD * (source[1]+1)**2
                    elif action > 8 and self.state_obj[dest[0], dest[1], dest[2]] == 0 and complete: # Place incorrect block
                        reward -= PLACE_REWARD
                    elif action < 9 and self.state_obj[source[0], source[1], source[2]] == 0 and complete: # Remove incorrect block
                        reward += PLACE_REWARD




        # Done?
        done, is_built = self.world.done(self.state_obj)
    
        # Perform observation
        s_sur , s_top ,_ = self._observe(agent_id,window,is_built) # ORIGINAL 5-TENSOR STATE
        
        
        self.finished |= done
        if initDone != self.finished:
            assert(self.finalAgentID == 0)
            self.finalAgentID = agent_id

        # Additional info
        info = self._listNextValidActions(agent_id, action)
        has_block = self.world.getBlock(self.world.getPos(agent_id) + np.array([0,1,0])) == -1

        # Unlock mutex
        self.mutex.release()
        

        return s_sur , s_top, reward, done, info, has_block, is_built

    def _getReward(self, reward_factor = 0.02):
        # Calculate number of correct/incorrect blocks
        good_blocks = 0
        for pos in PLAN_MAPS[self.map_id]:
            good_blocks += int(self.world.state[pos[0], pos[1], pos[2]] == -1) * (pos[1]+1)**2 # Squaring encourages the creation of ramps

        extra_blocks = (1 + np.clip(self.state_obj, -1, 0)) * np.clip(np.array(self.world.state), -1, 0) # Clip removes agent and sources
        bad_blocks = abs(np.sum(extra_blocks))

        assert good_blocks >= 0
        assert bad_blocks >= 0
        return (good_blocks - reward_factor * bad_blocks)

    def _listNextValidActions(self, agent_id, prev_action=0):
        available_actions = [] # NOP always allowed

        # Get current agent state
        agent_pos = self.world.getPos(agent_id)

        for action in range(1,5):     # Move
            validAction = False

            # Get estimated new agent state
            new_agent_pos = self.world.act(action, agent_id)

            # get coordinates and blocks near new position
            new_pos = new_agent_pos
            new_pos_upper = new_pos + np.array([0,1,0])
            new_pos_lower = new_pos + np.array([0,-1,0])
            new_pos_lower2 = new_pos + np.array([0,-2,0])

            block_newpos = self.world.getBlock(new_pos)
            block_upper = self.world.getBlock(new_pos_upper)
            block_lower = self.world.getBlock(new_pos_lower)
            block_lower2 = self.world.getBlock(new_pos_lower2)

            # execute movement if valid
            if block_newpos == 0:  # air in front?
                if block_lower == -1 or block_lower == -3:    # block or ground beneath?
                    dest = np.array(new_pos, dtype=int)
                    if (self.world.state[dest[0], 0:dest[1], dest[2]] == -1).all():
                        new_agent_pos = new_pos    # horizontal movement
                        validAction = True
                elif block_lower == 0 and block_lower2 in [-1, -3]:   # block or ground beneath?
                    dest = np.array(new_pos_lower, dtype=int)
                    if (self.world.state[dest[0], 0:dest[1], dest[2]] == -1).all():
                        new_agent_pos = new_pos_lower  # downstairs movement
                        validAction = True
            elif block_newpos == -1 and block_upper == 0:   # block in front and air above?
                dest = np.array(new_pos_upper, dtype=int)
                if (self.world.state[dest[0], 0:dest[1], dest[2]] == -1).all():
                    new_agent_pos = new_pos_upper    #upstairs movement
                    validAction = True

            # Prevent agents from accessing the highest level
            if new_agent_pos[1] == self.world_shape[1]-1:
                validAction = False

            if validAction:
                available_actions.append(action)

        for action in range(5,13):       # Pick & Place
            # determine block movement
            top = agent_pos + np.array([0,1,0])
            front = agent_pos + self.world.heading2vec((action-1) % 4)

            if action < 9:
                source = front
                dest = top
            else:
                source = top
                dest = front
            above_source = source + np.array([0,1,0])
            dest = np.array(dest, dtype=int)
            source = np.array(source, dtype=int)

            # execute
            if self.world.getBlock(source) in [-1, -2] and self.world.getBlock(dest) in [0, -2] and self.world.getBlock(above_source) in [0, -3] and (action < 9 or (action > 8 and (self.world.state[dest[0], 0:dest[1], dest[2]] == -1).all())) and not (action < 9 and source[1] == self.world_shape[1]-1):
                available_actions.append(action)

        if len(available_actions) > 1 and opposite_actions[prev_action] in available_actions:
            available_actions.remove(opposite_actions[prev_action])
        elif len(available_actions) == 0: # Only allow NOP if nothing else is valid
            available_actions.append(0)

        return available_actions
    
    # Render gridworld state
    def _render(self, agent_id=1, mode='human', close=False):
        world = self.world  # world = self.world.getObservation(agent_pos, self.ob_range)
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        depth = self.world.shape[1]
        min_size = 10 # minimum radius of the smallest square
        screen_width = 500
        screen_height = 500
        square_width = screen_width / world.shape[0]
        square_height = screen_height / world.shape[2]
        min_size = min(min_size, min(square_width, square_height))
        square_width_offset = (square_width-min_size) / depth
        square_height_offset = (square_height-min_size) / depth

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width,screen_height)
            self.squares = [[[rendering.FilledPolygon([(i*square_width+square_width_offset*k,(j+1)*square_height-square_height_offset*k),
                                                      (i*square_width+square_width_offset*k,j*square_height+square_height_offset*k),
                                                      ((i+1)*square_width-square_width_offset*k, j*square_height+square_height_offset*k),
                                                      ((i+1)*square_width-square_height_offset*k, (j+1)*square_height-square_height_offset*k)])
                              for k in range(self.world.shape[1])]
                            for i in range(self.world.shape[0])]
                           for j in range(self.world.shape[2])]
            for row in self.squares:
                for square in row:
                    for subsquare in square:
                        self.viewer.add_geom(subsquare)


        if self.world.state is None: return None

        for x in range(world.shape[0]):
            for y in range(world.shape[2]):
                for z in reversed(range(world.shape[1])):
                    val = world.getBlock([x, z, y])
                    new_color = AIR
                    if val == -2: # block spawn
                        new_color = BLOCK_SPAWN
                    elif val == -1:
                        new_color = BLOCK
                    elif val > 0:
                        new_color = AGENT
                    elif val != 0:
                        print('Error in map at {},{},{}, val = {}'.format(x,z,y,val))

                    if self.state_obj[x,z,y] == -1:
                        new_color = (new_color*4+PLAN_COLOR) / 5
                    if val == 0 and z != 0:
                        if self.state_obj[x,z,y] == -1:
                            self.squares[y][x][z]._color.vec4 = (new_color[0], new_color[1], new_color[2], 0.5)
                        else:
                            self.squares[y][x][z]._color.vec4 = (0,0,0,0)
                    else:
                        self.squares[y][x][z].set_color(*(new_color))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')





def make_gif(images, fname, duration=2, true_image=False,salience=False,salIMGS=None):
    imageio.mimwrite(fname,images,subrectangles=True)


# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def good_discount(x, gamma):
    return discount(x,gamma)

#Used to initialize weights for policy and value output layers (Do we need to use that? Maybe not now)
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer



class ACNet:
    def __init__(self, scope, a_size, trainer):
        
        with tf.variable_scope(str(scope)+'/qvalues'):
            self.is_Train = True
            #The input size may require more work to fit the interface.
            
            self.inputs_sur = tf.placeholder(shape=[1,4,120,160,3], dtype=tf.float32)
            self.inputs_top = tf.placeholder(shape=[1,1,120,160,3], dtype=tf.float32)
            self.policy, self.value, self.state_out, self.state_in, self.state_init, self.has_block, _ , self.is_built, self.position= self._build_net(self.inputs_sur, self.inputs_top)
            
        with tf.variable_scope(str(scope)+'/qvaluesB'):
#             self.inputsB = tf.placeholder(shape=[EXPERIENCE_BUFFER_SIZE,5,120,160,3], dtype=tf.float32)
            self.inputs_surB = tf.placeholder(shape=[EXPERIENCE_BUFFER_SIZE,4,120,160,3], dtype=tf.float32)
            self.inputs_topB = tf.placeholder(shape=[EXPERIENCE_BUFFER_SIZE,1,120,160,3], dtype=tf.float32)
            self.policyB, self.valueB, self.state_outB, self.state_inB, self.state_initB, self.has_blockB, self.validsB, self.is_builtB, self.positionB= self._build_net(self.inputs_surB, self.inputs_topB)
        if(scope!=GLOBAL_NET_SCOPE):
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
            self.valids = tf.placeholder(shape=[None,a_size], dtype=tf.float32)
            self.target_v = tf.placeholder(tf.float32, [None], 'Vtarget')
            self.target_has = tf.placeholder(tf.float32, [None])
            self.target_built = tf.placeholder(tf.float32, [None])
            self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)
            self.responsible_outputs = tf.reduce_sum(self.policyB * self.actions_onehot, [1])
            self.train_value = tf.placeholder(tf.float32, [None])
            self.target_pos =  tf.placeholder(tf.float32,[None,3])

            # Loss Functions
            self.value_loss = 0.5 * tf.reduce_sum(self.train_value*tf.square(self.target_v - tf.reshape(self.valueB, shape=[-1])))

            # something to encourage exploration
            self.entropy       = - tf.reduce_sum(self.policyB * tf.log(tf.clip_by_value(self.policyB,1e-10,1.0)))
            self.block_loss    = - tf.reduce_sum(self.target_has*tf.log(tf.clip_by_value(tf.reshape(self.has_blockB, shape=[-1]),1e-10,1.0))+(1-self.target_has)*tf.log(tf.clip_by_value(1-tf.reshape(self.has_blockB, shape=[-1]),1e-10,1.0)))
            self.built_loss    = - tf.reduce_sum(self.target_built*tf.log(tf.clip_by_value(tf.reshape(self.is_builtB, shape=[-1]),1e-10,1.0))+(1-self.target_built)*tf.log(tf.clip_by_value(1-tf.reshape(self.is_builtB, shape=[-1]),1e-10,1.0)))
            self.policy_loss   = - tf.reduce_sum(tf.log(tf.clip_by_value(self.responsible_outputs,1e-15,1.0)) * self.advantages)
            self.valid_loss    = - tf.reduce_sum(tf.log(tf.clip_by_value(self.validsB,1e-10,1.0)) * self.valids+tf.log(tf.clip_by_value(1-self.validsB,1e-10,1.0)) * (1-self.valids))
            self.position_loss = tf.sqrt(tf.reduce_sum(tf.square(self.target_pos-self.positionB)))
            self.loss = 0.5 * self.value_loss + self.policy_loss + 0.5*self.block_loss + 0.5*self.built_loss + 0.5*self.valid_loss - self.entropy * 0.01+ 0.5*self.position_loss

            # Get gradients from local network using local losses and
            # normalize the gradients using clipping
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope+'/qvaluesB')
            self.gradients = tf.gradients(self.loss, local_vars)
            self.var_norms = tf.global_norm(local_vars)
            grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, GRAD_CLIP)

            # Apply local gradients to global network
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, GLOBAL_NET_SCOPE+'/qvaluesB')
            self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))

            self.homogenize_weights = update_target_graph(str(scope)+'/qvaluesB', str(scope)+'/qvalues')
        if TRAINING:
            print("Hello World... From  "+str(scope))     # :)
                
    def conv_net(self,image):
        w_init= layers.variance_scaling_initializer()
        conv1 = layers.conv2d(inputs=image, padding="SAME", num_outputs=32, kernel_size=[6,8], stride=4, data_format="NHWC", weights_initializer=w_init,activation_fn=tf.nn.relu)
        max1  = layers.max_pool2d(inputs=conv1,kernel_size=[2,2],stride=2,data_format="NHWC")
        conv2 = layers.conv2d(inputs=max1, padding="SAME", num_outputs=64, kernel_size=[6,8], stride=2, data_format="NHWC", weights_initializer=w_init,activation_fn=tf.nn.relu)
        max2  = layers.max_pool2d(inputs=conv2,kernel_size=[2,2],stride=2,data_format="NHWC")
        conv3 = layers.conv2d(inputs=max2, padding="VALID", num_outputs=64, kernel_size=[3,4], stride=2, data_format="NHWC", weights_initializer=w_init,activation_fn=tf.nn.relu)
        flat  = layers.flatten(conv3)
        flat  =  tf.expand_dims(flat,axis=1)
        
        return flat
    
    
    

    def _build_net(self,inputs_sur, inputs_top):

        similarity_sur , similarity_up = [], []
        # similarity = np.zeros([inputs_.shape[0],inputs_.shape[1],num_out])
        for i in range(inputs_sur.shape[1]):
        
            flat = self.conv_net(inputs_sur[:,i,:,:,:])
            if i==0:
                similarity_sur = flat
            else:
                similarity_sur = tf.concat([similarity_sur, flat], 1)
                               
        for i in range(inputs_top.shape[1]):

            flat = self.conv_net(inputs_top[:,i,:,:,:])

        similarity_up = flat

        similarity = tf.concat([similarity_sur,similarity_up], 1)
                
        flat_ = layers.flatten(similarity)
        

        f2 = layers.fully_connected(inputs=flat_,  num_outputs=512)
        h3 = layers.fully_connected(inputs=f2,  num_outputs=512)
        

        #Recurrent network for temporal dependencies
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(512,state_is_tuple=True)
        c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
        h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
        state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
        state_in = (c_in, h_in)
        rnn_in = tf.expand_dims(h3, [0])
        step_size = tf.shape(inputs_sur)[:1]
        state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
        lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
        time_major=False)
        lstm_c, lstm_h = lstm_state
        state_out = (lstm_c[:1, :], lstm_h[:1, :])
        rnn_out = tf.reshape(lstm_outputs, [-1, 512])

        policy_layer = layers.fully_connected(inputs=rnn_out, num_outputs=a_size,weights_initializer=normalized_columns_initializer(1./float(a_size)), biases_initializer=None, activation_fn=None)
        policy=tf.nn.softmax(policy_layer)
        policy_sig=tf.sigmoid(policy_layer)
        value = layers.fully_connected(inputs=rnn_out, num_outputs=1,weights_initializer=normalized_columns_initializer(1.0), biases_initializer=None, activation_fn=None)
        has_block = layers.fully_connected(inputs=rnn_out, num_outputs=1,weights_initializer=normalized_columns_initializer(1.0), biases_initializer=None, activation_fn=tf.sigmoid)
        is_built  = layers.fully_connected(inputs=rnn_out, num_outputs=1,weights_initializer=normalized_columns_initializer(1.0), biases_initializer=None, activation_fn=tf.sigmoid)
        position  = layers.fully_connected(inputs=rnn_out, num_outputs=3,weights_initializer=normalized_columns_initializer(1.0), biases_initializer=None, activation_fn=None)
        

        return policy, value, state_out ,state_in, state_init, has_block, policy_sig, is_built, position

class Worker:
    def __init__(self, game, metaAgentID, workerID, a_size, groupLock):
        self.workerID = workerID
        self.env = game
        self.metaAgentID = metaAgentID
        self.name = "worker_"+str(workerID)
        self.agentID = ((workerID-1) % num_workers) + 1 
        self.groupLock = groupLock
        self.window = []
        
        self.nextGIF = episode_count # For GIFs output
        #Create the local copy of the network and the tensorflow op to copy global parameters to local network
        self.local_AC = ACNet(self.name,a_size,trainer)
        self.copy_weights = self.local_AC.homogenize_weights
        self.pull_global = update_target_graph(GLOBAL_NET_SCOPE, self.name)
        
    def save_pics(self,step,ScreenShots):
        print("Saving pics {}".format(step))
        for i in range(5):
            scipy.misc.imsave('metaagent {} agent {} step {} image{}.jpg'.format(self.metaAgentID,self.agentID,step,i), ScreenShots[i])
        
    def train(self, rollout, sess, gamma, bootstrap_value):
#       [s,a,r,s1,d,v[0,0],train_valid,pred_has_block,int(has_block),train_val,int(is_built),real_pos,pred_pos]
# s_sur , s_top,a,r,s_sur1, s_top1,d,v[0,0],train_valid,pred_has_block,int(has_block),train_val,int(is_built)] + real_pos
        rollout = np.array(rollout)
    
        observations_sur = rollout[:,0]
        observations_top = rollout[:,1]
        actions = rollout[:,2]
        rewards = rollout[:,3]
        lastDone = rollout[-1,6]
        values = rollout[:,7]
        valids = rollout[:,8]
        pred_has = rollout[:,9]
        has_blocks = rollout[:,10]
        train_value = rollout[:,11]
        is_built = rollout[:,12]
        real_position = rollout[:,[13,14,15]]

        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. (With bootstrapping)
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = good_discount(advantages,gamma)

#         if not lastDone:
        num_samples = min(EPISODE_SAMPLES,len(advantages))
        sampleInd = np.sort(np.random.choice(advantages.shape[0], size=(num_samples,), replace=False))

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        rnn_state = self.local_AC.state_init
        feed_dict = {self.local_AC.target_v:np.stack(discounted_rewards),
#            self.local_AC.inputsB:np.squeeze(np.stack(observations), axis=1),
            self.local_AC.inputs_surB:np.stack(observations_sur),
            self.local_AC.inputs_topB:np.stack(observations_top),
            self.local_AC.actions:actions,
            self.local_AC.valids:np.stack(valids),
            self.local_AC.advantages:advantages,
            self.local_AC.train_value:train_value,
            self.local_AC.has_blockB:np.reshape(pred_has,[np.shape(pred_has)[0],1]),
            self.local_AC.target_has:has_blocks,
            self.local_AC.target_built:is_built,
            self.local_AC.state_inB[0]:rnn_state[0],
            self.local_AC.state_inB[1]:rnn_state[1],
            self.local_AC.target_pos:real_position}

        v_l,p_l,b_l,valid_l,bp_l,pos_l,e_l,g_n,v_n,_ = sess.run([self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.block_loss,
            self.local_AC.valid_loss,
            self.local_AC.built_loss,
            self.local_AC.position_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)

        return v_l / len(rollout), p_l / len(rollout), b_l / len(rollout), valid_l/len(rollout), bp_l / len(rollout), pos_l / len(rollout), e_l / len(rollout), g_n, v_n

    def shouldRun(self, coord, episode_count):
        if TRAINING:
            return (not coord.should_stop())
        else:
            return (episode_count < NUM_EXPS)

    def work(self,max_episode_length,gamma,sess,coord,saver):
        global episode_count, episode_rewards, episode_lengths, episode_mean_values, episode_invalid_ops, started_threads, not_loaded_model, NUM_META_AGENTS, NUM_THREADS, WINDOWS
        total_steps = 0
        
        # The same NUM_THREADS should have access to the same window

        with sess.as_default(), sess.graph.as_default():
            while self.shouldRun(coord, episode_count):
                
                sess.run(self.pull_global)
                sess.run(self.copy_weights)

                episode_buffer, episode_values = [], []
                episode_reward = episode_step_count = episode_inv_count = 0
                if not TRAINING:
                    completedQ[self.metaAgentID], d = False, False
                    completed_time[self.metaAgentID] = np.nan
                    
                # Initial state from the environment
                if FULL_PLAN and random.random() < 0.5:
                    validActions,has_block,is_built = self.env._reset(self.agentID, empty=EMPTY, full=True)
                else:
                    validActions,has_block,is_built = self.env._reset(self.agentID, empty=EMPTY, full=False)
                    
                if episode_count==0 or (load_model==True and not_loaded_model!= NUM_META_AGENTS*NUM_THREADS):

                    print("Creating window for MetaAgent={} and NUM_THREADS= {}".format(self.metaAgentID,self.agentID))
                    self.window = Window(width=160, height=120, caption='Pyglet', resizable=True, visible=True)
                    started_thread[self.metaAgentID*NUM_THREADS+self.agentID-1]= True
                    not_loaded_model += 1
                    
                s_sur , s_top, real_pos= self.env._observe(self.agentID,self.window,is_built,self.metaAgentID)
#                 self.save_pics(episode_step_count,s)
                
                rnn_state = self.local_AC.state_init
                RewardNb = wrong_block = wrong_built = 0

                saveGIF = False
                if OUTPUT_GIFS and self.workerID == 1 and ((not TRAINING) or (episode_count >= self.nextGIF and episode_count >= OBSERVE_PERIOD)):
                    saveGIF = True
                    self.nextGIF += 128
                    GIF_episode = int(episode_count)
                    episode_frames = [ self.env._render(mode='rgb_array') ]

                self.groupLock.release(0,self.name)
                self.groupLock.acquire(1,self.name) # synchronize starting time of the threads

                print("({}) Starting Episode {}...".format(self.workerID, episode_count))
                while (not self.env.finished): # Give me something!

                    #Take an action using probabilities from policy network output
                    a_dist,v,rnn_state,pred_has_block,pred_is_built,pred_pos = sess.run([self.local_AC.policy,self.local_AC.value,self.local_AC.state_out,self.local_AC.has_block,self.local_AC.is_built,self.local_AC.position], feed_dict={self.local_AC.inputs_sur:[s_sur],self.local_AC.inputs_top:[s_top],self.local_AC.state_in[0]:rnn_state[0],self.local_AC.state_in[1]:rnn_state[1]})
                    
                
                    if(not (np.argmax(a_dist.flatten()) in validActions)):
                        episode_inv_count+=1
                    train_valid=np.zeros(a_size)
                    train_valid[validActions]=1
                    mysum=np.sum(train_valid)

                    valid_dist = np.array([a_dist[0,validActions]])
                    valid_dist /= np.sum(valid_dist)
                    if TRAINING:
                        if  (pred_has_block.flatten()[0] < 0.5) == has_block:
                            wrong_block += 1
                            a = np.random.choice(validActions)
                            train_val = 0
                        elif (pred_is_built.flatten()[0] < 0.5) == is_built:
                            wrong_built += 1
                            a = validActions[ np.random.choice(range(valid_dist.shape[1]),p=valid_dist.ravel()) ]
                            train_val = 1.
                        else:
                            a = validActions[ np.random.choice(range(valid_dist.shape[1]),p=valid_dist.ravel()) ]
                            train_val = 1.
                    else:
                        if GREEDY:
                            a = np.argmax(a_dist.flatten())
                        if not GREEDY or a not in validActions:
                            a = validActions[ np.random.choice(range(valid_dist.shape[1]),p=valid_dist.ravel()) ]
                        train_val = 1.
                    
                    #taking action
                    _, _, r, d, validActions, has_block1, is_built1 = self.env.step((self.agentID,a,self.window))


                    if not TRAINING:
                        extraBlocks = max(0, self.env.world.countExtraBlocks(self.env.state_obj))
                        if np.isnan(completed_time[self.metaAgentID]) and completedQ[self.metaAgentID] != is_built1 and is_built1:
                            completed_time[self.metaAgentID] = episode_step_count+1
                            scaffoldings[self.metaAgentID]   = extraBlocks
                            blocks_left[self.metaAgentID]    = extraBlocks
                        elif is_built1 and not np.isnan(blocks_left[self.metaAgentID]) and extraBlocks < blocks_left[self.metaAgentID]:
                            blocks_left[self.metaAgentID]    = extraBlocks
                        completedQ[self.metaAgentID]        |= is_built1

                    self.groupLock.release(1,self.name)
                    self.groupLock.acquire(0,self.name) # synchronize threads

                    # Get common observation for all agents after all individual actions have been performed
                    s_sur1 , s_top1, real_pos1= self.env._observe(self.agentID,self.window,is_built,self.metaAgentID)
#                     self.save_pics(episode_step_count,s1)
                    d  = self.env.finished

                    if saveGIF:
                        episode_frames.append(self.env._render(mode='rgb_array'))

                    self.groupLock.release(0,self.name)
                    self.groupLock.acquire(1,self.name) # synchronize threads

                    episode_buffer.append([s_sur , s_top,a,r,s_sur1, s_top1,d,v[0,0],train_valid,pred_has_block,int(has_block),train_val,int(is_built)] + real_pos)
                    episode_values.append(v[0,0])
                    episode_reward += r
                    real_pos = real_pos1
                    s_sur = s_sur1
                    s_top = s_top1
                    total_steps += 1
                    has_block = has_block1
                    is_built = is_built1
                    episode_step_count += 1

                    if r>0:
                        RewardNb += 1
                    if d == True and TRAINING:
                        print('\n{} Goodbye World. We did it!'.format(episode_step_count))

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if TRAINING and (len(episode_buffer) % EXPERIENCE_BUFFER_SIZE == 0 or d):
                        # Since we don't know what the true final return is, we "bootstrap" from our current value estimation.
                        if len(episode_buffer) >= EXPERIENCE_BUFFER_SIZE:

                            if d:
                                s1Value = 0.0
                            else:
                                s1Value = sess.run(self.local_AC.value,
                                     feed_dict={self.local_AC.inputs_sur:[s_sur],
                                                self.local_AC.inputs_top:[s_top],
                                                self.local_AC.state_in[0]:rnn_state[0],
                                                self.local_AC.state_in[1]:rnn_state[1]})[0,0]

                            v_l,p_l,b_l,valid_l,bp_l,pos_l,e_l,g_n,v_n = self.train(episode_buffer[-EXPERIENCE_BUFFER_SIZE:],sess,gamma,s1Value)

                            sess.run(self.pull_global)
                            sess.run(self.copy_weights)

                    if episode_step_count >= max_episode_length or d:
                        break

                episode_rewards[self.metaAgentID].append(episode_reward)
                episode_lengths[self.metaAgentID].append(episode_step_count)
                episode_mean_values[self.metaAgentID].append(np.nanmean(episode_values))
                episode_invalid_ops[self.metaAgentID].append(episode_inv_count)

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % EXPERIENCE_BUFFER_SIZE == 0 and printQ:
                    print('{} Episode terminated ({},{})'.format(episode_count, self.agentID, RewardNb))

                    
                if TRAINING:
                    episode_count += 1. / num_workers

                    if episode_count % SUMMARY_WINDOW == 0:
                        if episode_count % 500 == 0:
                            saver.save(sess, model_path+'/model-'+str(int(episode_count))+'.cptk')
                            print ('Saved Model')
                        mean_reward = np.mean(episode_rewards[self.metaAgentID][-SUMMARY_WINDOW:])
                        mean_length = np.mean(episode_lengths[self.metaAgentID][-SUMMARY_WINDOW:])
                        mean_value = np.mean(episode_mean_values[self.metaAgentID][-SUMMARY_WINDOW:])
                        mean_invalid = np.mean(episode_invalid_ops[self.metaAgentID][-SUMMARY_WINDOW:])
                        current_learning_rate = sess.run(lr,feed_dict={global_step:episode_count})

                        summary = tf.Summary()
                        summary.value.add(tag='Perf/Learning Rate',simple_value=current_learning_rate)
                        summary.value.add(tag='Perf/Reward', simple_value=mean_reward)
                        summary.value.add(tag='Perf/Length', simple_value=mean_length)
                        summary.value.add(tag='Perf/Valid Rate', simple_value=(mean_length-mean_invalid)/mean_length)
                        summary.value.add(tag='Perf/Block Prediction Accuracy', simple_value=float(episode_step_count-wrong_block)/float(episode_step_count))
                        summary.value.add(tag='Perf/Plan Completion Accuracy', simple_value=float(episode_step_count-wrong_built)/float(episode_step_count))

                        summary.value.add(tag='Losses/Value Loss', simple_value=v_l)
                        summary.value.add(tag='Losses/Policy Loss', simple_value=p_l)
                        summary.value.add(tag='Losses/Plan Completion Loss', simple_value=bp_l)
                        summary.value.add(tag='Losses/Block Prediction Loss', simple_value=b_l)
                        summary.value.add(tag='Losses/Valid Loss', simple_value=valid_l)
                        summary.value.add(tag='Losses/Grad Norm', simple_value=g_n)
                        summary.value.add(tag='Losses/Var Norm', simple_value=v_n)
                        summary.value.add(tag='Losses/Position Loss', simple_value=pos_l)
                        global_summary.add_summary(summary, int(episode_count))

                        global_summary.flush()

                        if printQ:
                            print('{} Tensorboard updated ({})'.format(episode_count, self.workerID))
                elif not TRAINING and self.workerID == 1:
                    if episode_buffer[0][-1] == 0: # only care about plan completion if init state didn't contain the completed plan...
                        completed[episode_count]      = int(completedQ[self.metaAgentID])
                    if not np.isnan(completed_time[self.metaAgentID]):
                        plan_durations[episode_count] = completed_time[self.metaAgentID]
                        rollout = np.array(episode_buffer)
                        place_moves[episode_count]    = np.sum( np.asarray(rollout[:completed_time[self.metaAgentID]+1,1] > 8, dtype=int) )
                    len_episodes[episode_count]       = episode_step_count
#                     saveGIF &= (episode_step_count < max_episode_length)

                    if not np.isnan(completed_time[self.metaAgentID]):
                        episode_count += 1
                    GIF_episode = int(episode_count)
#                     print('({}) Thread {}: {} steps ({} invalids).'.format(episode_count, self.workerID, episode_step_count, episode_inv_count))

                self.groupLock.release(1,self.name)
                self.groupLock.acquire(0,self.name) # synchronize threads

                if saveGIF:
                    # Dump episode frames for external gif generation (otherwise, makes the jupyter kernel crash)
                    time_per_step = 0.1
                    images = np.array(episode_frames)
                    if TRAINING:
                        gif_creation = lambda: make_gif(images, '{}/episode_{:d}_{:d}_{:.1f}.gif'.format(gifs_path,GIF_episode,episode_step_count,episode_reward), duration=len(images)*time_per_step,true_image=True,salience=False)
                        threading.Thread(target=(gif_creation)).start()
                    else:
                        make_gif(images, '{}/episode_{:d}_{:d}.gif'.format(gifs_path,GIF_episode,episode_step_count), duration=len(images)*time_per_step,true_image=True,salience=False)
                if self.workerID == 1 and SAVE_EPISODE_BUFFER and episode_step_count < max_episode_length:
                    with open('{}/episode_{}.dat'.format(episodes_path,GIF_episode), 'wb') as file:
                        pickle.dump(episode_buffer, file)


# Learning parameters
max_episode_length     = 512
episode_count          = 0
EPISODE_START          = episode_count
gamma                  = .9 # discount rate for advantage estimation and reward discounting
GRAD_CLIP              = 300.0
LR_Q                   = 2.e-5 
ADAPT_LR               = False
ADAPT_COEFF            = 1.e-3 / 20. #the coefficient A in LR_Q/sqrt(A*steps+1) for calculating LR
a_size                 = 13 
EXPERIENCE_BUFFER_SIZE = 256
OBSERVE_PERIOD         = 0. # Period of pure observation (no value learning)
SUMMARY_WINDOW         = 25
NUM_META_AGENTS        = 2
NUM_THREADS            = 4 #int(multiprocessing.cpu_count() / (2 * NUM_META_AGENTS))
EPISODE_SAMPLES        = EXPERIENCE_BUFFER_SIZE
load_model             = False
RESET_TRAINER          = False
model_path             = './model_Vision' #location for saving models
gifs_path              = './gifs_Vision'
train_path             = 'train_Vision' #location for saving tensorboard files
episodes_path          = 'gifs3D' # used to export episode_buffers that can be read/played/recorded by the visualizer
GLOBAL_NET_SCOPE       = 'global'

# Simulation options
FULL_HELP              = False
MAP_ID                 = 0 #0: RANDOMIZED_PLAN, other: given map (list in minecraft_SA4H.py)
OUTPUT_GIFS            = False
SAVE_EPISODE_BUFFER    = False

#Goal Images
NotProcessed           = True           

# Testing
TRAINING               = True
GREEDY                 = False
NUM_EXPS               = 20
EMPTY                  = True and (not TRAINING)
FULL_PLAN              = False # Help training cleanup by forcing all episodes
MODEL_NUMBER           = 0 # to start with the structure already completed
                               # Should be enabled near the end of training
started_thread         = []
not_loaded_model       = 0

# Shared arrays for tensorboard
episode_rewards        = [ []     for _ in range(NUM_META_AGENTS) ]
episode_lengths        = [ []     for _ in range(NUM_META_AGENTS) ]
episode_mean_values    = [ []     for _ in range(NUM_META_AGENTS) ]
episode_invalid_ops    = [ []     for _ in range(NUM_META_AGENTS) ]
completedQ             = [ False  for _ in range(NUM_META_AGENTS) ]
completed_time         = [ np.nan for _ in range(NUM_META_AGENTS) ]
printQ                 = False # (for headless)



tf.reset_default_graph()
print("Hello World")
if not os.path.exists(model_path):
    os.makedirs(model_path)
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth=True

if not TRAINING:
    blocks_left    = np.array([np.nan for _ in range(NUM_EXPS)])
    scaffoldings   = np.array([np.nan for _ in range(NUM_EXPS)])
    completed      = np.array([np.nan for _ in range(NUM_EXPS)])
    plan_durations = np.array([np.nan for _ in range(NUM_EXPS)])
    len_episodes   = np.array([np.nan for _ in range(NUM_EXPS)])
    place_moves    = np.array([np.nan for _ in range(NUM_EXPS)]) # Only makes sense for a single agent (comparison with TERMES code)
    mutex = threading.Lock()
    gifs_path += '_tests'
    if SAVE_EPISODE_BUFFER and not os.path.exists(episodes_path):
        os.makedirs(episodes_path)

#Create a directory to save episode playback gifs to
if OUTPUT_GIFS and not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


    
    
started_thread = []
for i in range(NUM_META_AGENTS):
    for j in range(NUM_THREADS):
        start = False
        started_thread.append(start)
    
with tf.device("/gpu:0"):
    master_network = ACNet(GLOBAL_NET_SCOPE,a_size,None) # Generate global network
    global_step = tf.placeholder(tf.float32)
    if ADAPT_LR:
        #computes LR_Q/sqrt(ADAPT_COEFF*steps+1)
        #we need the +1 so that lr at step 0 is defined
        lr=tf.divide(tf.constant(LR_Q),tf.sqrt(tf.add(1.,tf.multiply(tf.constant(ADAPT_COEFF),global_step))))
    else:
        lr=tf.constant(LR_Q)
    trainer = tf.contrib.opt.NadamOptimizer(learning_rate=lr, use_locking=True)

    num_workers = NUM_THREADS # Set workers # = # of available CPU threads
    if not TRAINING:
        NUM_META_AGENTS = 1

    gameEnvs, workers, groupLocks = [], [], []
    for ma in range(NUM_META_AGENTS):
        gameEnv = MinecraftEnv(num_workers, observation_range=-1, observation_mode='default', FULL_HELP=FULL_HELP, MAP_ID=MAP_ID)
        gameEnvs.append(gameEnv)

        # Create groupLock
        workerNames = ["worker_"+str(i) for i in range(ma*num_workers+1,(ma+1)*num_workers+1)]
        groupLock = GroupLock.GroupLock([workerNames,workerNames])
        groupLocks.append(groupLock)

        # Create worker classes
        workersTmp = []
        for i in range(ma*num_workers+1,(ma+1)*num_workers+1):
            workersTmp.append(Worker(gameEnv,ma,i,a_size,groupLock))
        workers.append(workersTmp)

    if TRAINING:
        global_summary = tf.summary.FileWriter(train_path)
    else:
        global_summary = 0
    saver = tf.train.Saver(max_to_keep=0)

    with tf.Session(config=config) as sess:
        coord = tf.train.Coordinator()
        if load_model == True:
            print ('Loading Model...')
            if not TRAINING:
                with open(model_path+'/checkpoint', 'w') as file:
                    file.write('model_checkpoint_path: "model-{}.cptk"'.format(MODEL_NUMBER))
                    file.close()
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess,ckpt.model_checkpoint_path)
            if RESET_TRAINER:
                trainer = tf.contrib.opt.NadamOptimizer(learning_rate=lr, use_locking=True)
        else:
            sess.run(tf.global_variables_initializer())

        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate thread.
        worker_threads = []
        for ma in range(NUM_META_AGENTS):
            NT = 0
            for worker in workers[ma]:
                groupLocks[ma].acquire(0,worker.name) # synchronize starting time of the threads
                worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
                print("Starting worker " + str(worker.workerID))
                t = threading.Thread(target=(worker_work))
                t.start()
                while started_thread[worker.metaAgentID*NUM_THREADS+worker.agentID-1]== False:
                    continue
                worker_threads.append(t)
                NT += 1
        coord.join(worker_threads)
        
        print("JOIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIINNNNNNNNNNNNNNNNNNNNEEEEEEEEEEEEEEEEEEEEEEEEDDDDDD")


if not TRAINING:
    print('[{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}]'.format(
           1 - np.nanmean(blocks_left/scaffoldings), np.sqrt(np.nanvar(blocks_left/scaffoldings)),
           np.nanmean(completed), np.sqrt(np.nanvar(completed)),
           np.nanmean(plan_durations), np.sqrt(np.nanvar(plan_durations)),
           np.nanmean(len_episodes), np.sqrt(np.nanvar(len_episodes)),
           np.nanmean(np.asarray(len_episodes < max_episode_length, dtype=float)),
           np.nanmean(blocks_left), np.sqrt(np.nanvar(blocks_left)),
           np.nanmean(scaffoldings), np.sqrt(np.nanvar(scaffoldings)),
           np.nanmean(place_moves), np.sqrt(np.nanvar(place_moves)))
         )



# Validation code
"""
max_episode_length     = 2000
GREEDY                 = False
NUM_EXPS               = 100
EMPTY                  = True
TRAINING               = False

MODEL_NUMBER           = 67000
SAVE_EPISODE_BUFFER    = True
NUM_META_AGENTS        = 1

if SAVE_EPISODE_BUFFER and not os.path.exists('gifs3D'):
    os.makedirs('gifs3D')

for num_workers in [1]:#,2,4,8,12,16,3,6,10,14]:
    for MAP_ID in range(6,7):
        episode_count = 0

        tf.reset_default_graph()
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth=True

        blocks_left    = np.array([np.nan for _ in range(NUM_EXPS)])
        scaffoldings   = np.array([np.nan for _ in range(NUM_EXPS)])
        completed      = np.array([np.nan for _ in range(NUM_EXPS)])
        plan_durations = np.array([np.nan for _ in range(NUM_EXPS)])
        len_episodes   = np.array([np.nan for _ in range(NUM_EXPS)])
        place_moves    = np.array([np.nan for _ in range(NUM_EXPS)]) # Only makes sense for a single agent (comparison with TERMES code)
        mutex = threading.Lock()

        episodes_path = 'gifs3D/{:d}_{:d}'.format(num_workers,MAP_ID)
        if SAVE_EPISODE_BUFFER and not os.path.exists(episodes_path):
            os.makedirs(episodes_path)

        with tf.device("/gpu:0"):
            master_network = ACNet(GLOBAL_NET_SCOPE,a_size,None) # Generate global network
            trainer = tf.contrib.opt.NadamOptimizer(learning_rate=LR_Q, use_locking=True)

            global_summary = 0
            saver = tf.train.Saver(max_to_keep=5)

            with tf.Session(config=config) as sess:
                coord = tf.train.Coordinator()
                with open(model_path+'/checkpoint', 'w') as file:
                    file.write('model_checkpoint_path: "model-{}.cptk"'.format(MODEL_NUMBER))
                    file.close()
                ckpt = tf.train.get_checkpoint_state(model_path)
                saver.restore(sess,ckpt.model_checkpoint_path)

                gameEnvs, workers, groupLocks = [], [], []
                for ma in range(NUM_META_AGENTS):
                    gameEnv = minecraft.MinecraftEnv(num_workers, observation_range=-1, observation_mode='default', FULL_HELP=FULL_HELP, MAP_ID=MAP_ID)
                    gameEnvs.append(gameEnv)

                    # Create groupLock
                    workerNames = ["worker_"+str(i) for i in range(ma*num_workers+1,(ma+1)*num_workers+1)]
                    groupLock = GroupLock.GroupLock([workerNames,workerNames])
                    groupLocks.append(groupLock)

                    # Create worker classes
                    workersTmp = []
                    for i in range(ma*num_workers+1,(ma+1)*num_workers+1):
                        workersTmp.append(Worker(gameEnv,ma,i,a_size,groupLock))
                    workers.append(workersTmp)

                # This is where the asynchronous magic happens.
                # Start the "work" process for each worker in a separate thread.
                worker_threads = []
                for ma in range(NUM_META_AGENTS):
                    for worker in workers[ma]:
                        groupLocks[ma].acquire(0,worker.name) # synchronize starting time of the threads
                        worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
                        t = threading.Thread(target=(worker_work))
                        t.start()
                        worker_threads.append(t)
                coord.join(worker_threads)

        print('[{:d}, {:d}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}]'.format(
               num_workers, MAP_ID,
               1 - np.nanmean(blocks_left/scaffoldings), np.sqrt(np.nanvar(blocks_left/scaffoldings)),
               np.nanmean(completed), np.sqrt(np.nanvar(completed)),
               np.nanmean(plan_durations), np.sqrt(np.nanvar(plan_durations)),
               np.nanmean(len_episodes), np.sqrt(np.nanvar(len_episodes)),
               np.nanmean(np.asarray(len_episodes < max_episode_length, dtype=float)),
               np.nanmean(blocks_left), np.sqrt(np.nanvar(blocks_left)),
               np.nanmean(scaffoldings), np.sqrt(np.nanvar(scaffoldings)),
               np.nanmean(place_moves), np.sqrt(np.nanvar(place_moves)))
             )

        ofp = open('results.txt','a')
        ofp.write('{:d}, {:d}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}\n'.format(
               num_workers, MAP_ID,
               1 - np.nanmean(blocks_left/scaffoldings), np.sqrt(np.nanvar(blocks_left/scaffoldings)),
               np.nanmean(completed), np.sqrt(np.nanvar(completed)),
               np.nanmean(plan_durations), np.sqrt(np.nanvar(plan_durations)),
               np.nanmean(len_episodes), np.sqrt(np.nanvar(len_episodes)),
               np.nanmean(np.asarray(len_episodes < max_episode_length, dtype=float)),
               np.nanmean(blocks_left), np.sqrt(np.nanvar(blocks_left)),
               np.nanmean(scaffoldings), np.sqrt(np.nanvar(scaffoldings)),
               np.nanmean(place_moves), np.sqrt(np.nanvar(place_moves)))
             )
        ofp.close()
"""


