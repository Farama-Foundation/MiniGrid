from __future__ import annotations

import numpy as np

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava
from minigrid.minigrid_env import MiniGridEnv


class MazeEnv(MiniGridEnv):

    """
    ## Description

    Depending on the `obstacle_type` parameter:
    - `Lava` - The agent has to reach the green goal square on the other corner
        of the room while avoiding rivers of deadly lava which terminate the
        episode in failure. Each lava stream runs across the room either
        horizontally or vertically, and has a single crossing point which can be
        safely used; Luckily, a path to the goal is guaranteed to exist. This
        environment is useful for studying safety and safe exploration.
    - otherwise - Similar to the `LavaCrossing` environment, the agent has to
        reach the green goal square on the other corner of the room, however
        lava is replaced by walls. This MDP is therefore much easier and maybe
        useful for quickly testing your algorithms.

    ## Mission Space
    Depending on the `obstacle_type` parameter:
    - `Lava` - "avoid the lava and get to the green goal square"
    - otherwise - "find the opening and get to the green goal square"

    ## Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. The agent falls into lava.
    3. Timeout (see `max_steps`).

    ## Registered Configurations

    S: size of the map SxS.
    N: number of valid crossings across lava or walls from the starting position
    to the goal

    - `Lava` :
        - `MiniGrid-LavaCrossingS9N1-v0`
        - `MiniGrid-LavaCrossingS9N2-v0`
        - `MiniGrid-LavaCrossingS9N3-v0`
        - `MiniGrid-LavaCrossingS11N5-v0`

    - otherwise :
        - `MiniGrid-SimpleCrossingS9N1-v0`
        - `MiniGrid-SimpleCrossingS9N2-v0`
        - `MiniGrid-SimpleCrossingS9N3-v0`
        - `MiniGrid-SimpleCrossingS11N5-v0`

    """

    def __init__(
        self,
        size=9,
        obstacle_type=Lava,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.obstacle_type = obstacle_type

        if obstacle_type == Lava:
            mission_space = MissionSpace(mission_func=self._gen_mission_lava)
        else:
            mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=False,  # Set this to True for maximum speed
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission_lava():
        return "avoid the lava and get to the green goal square"

    @staticmethod
    def _gen_mission():
        return "find the opening and get to the green goal square"

    def _gen_grid(self, width, height):
        assert width % 2 == 1 and height % 2 == 1  # odd size

        # Create an empty grid
        self.grid = Grid(width, height)

        # cell obj
        cell = object()

        starting_height = max(1,int(self.np_random.random()*(height-1)))
        starting_width = max(1,int(self.np_random.random()*(width-1)))

        self.grid.set(starting_height, starting_width, cell)

        walls = []
        for i,j in ([-1,0],[0,-1],[0,1],[1,0]):
            walls.append([starting_height+i,starting_width+j])
            self.put_obj(self.obstacle_type(), starting_height+i, starting_width+j)

        # Find number of surrounding cells
        def surroundingCells(rand_wall):
            s_cells = 0
            for i,j in ([-1,0],[0,-1],[0,1],[1,0]):
                if (self.grid.get(rand_wall[0]+i,rand_wall[1]+j) == cell):
                    s_cells += 1
            return s_cells

        def delete_wall(walls, rand_wall):
            for wall in walls:
                if (wall[0] == rand_wall[0] and wall[1] == rand_wall[1]):
                    walls.remove(wall)
        
        def mark(walls, cell, p, x, y, bound):
            if (p != bound):
                if (self.grid.get(x, y) != cell):
                    self.put_obj(self.obstacle_type(), x, y)
                if ([x, y] not in walls):
                    walls.append([x, y])
        
        def helper(rand_wall, cell, wall1, wall2, wall3):
            # Find the number of surrounding cells
            s_cells = surroundingCells(rand_wall)
            if (s_cells < 2):
                # Denote the new path
                self.grid.set(rand_wall[0], rand_wall[1], cell)

                # Mark the new walls
                mark(*wall1)

                mark(*wall2)

                mark(*wall3)

        while walls:
                # Pick a random wall
                rand_wall = walls[int(self.np_random.random()*len(walls))-1]

                # Upper cell
                up = (walls, cell, rand_wall[0], rand_wall[0]-1, rand_wall[1], 0)

                # Bottom cell
                bot = (walls, cell, rand_wall[0], rand_wall[0]+1, rand_wall[1], height-1)

                # Leftmost cell
                left = (walls, cell, rand_wall[1], rand_wall[0], rand_wall[1]-1, 0)

                # Rightmost cell
                right = (walls, cell, rand_wall[1], rand_wall[0], rand_wall[1]+1, width-1)
            
                # Check if it is a left wall
                if (rand_wall[1] != 0):

                    if (self.grid.get(rand_wall[0],rand_wall[1]-1) is None and self.grid.get(rand_wall[0],rand_wall[1]+1) == cell):

                        helper(rand_wall, cell, up, bot, left)

                        # Delete wall
                        delete_wall(walls, rand_wall)

                        continue
                
                # Check if it is an upper wall
                if (rand_wall[0] != 0):
                    if (self.grid.get(rand_wall[0]-1,rand_wall[1]) is None and self.grid.get(rand_wall[0]+1,rand_wall[1]) == cell):
                        
                        helper(rand_wall, cell, up, left, right)

                        # Delete wall
                        delete_wall(walls, rand_wall)

                        continue
            

                # Check the bottom wall
                if (rand_wall[0] != height-1):
                    if (self.grid.get(rand_wall[0]+1,rand_wall[1]) is None and self.grid.get(rand_wall[0]-1,rand_wall[1]) == cell):

                        helper(rand_wall, cell, bot, left, right)

                        # Delete wall
                        delete_wall(walls, rand_wall)

                        continue

                # Check the right wall
                if (rand_wall[1] != width-1):
                    if (self.grid.get(rand_wall[0],rand_wall[1]+1) is None and self.grid.get(rand_wall[0],rand_wall[1]-1) == cell):

                        helper(rand_wall, cell, right, bot, up)

                        # Delete wall
                        delete_wall(walls, rand_wall)

                        continue

                # Delete the wall from the list anyway
                delete_wall(walls, rand_wall)

        # Set entrance and exit
        cells = []
        for i in range(0, width):
            if (self.grid.get(1,i) == cell):
                cells.append((1,i))
        
        self.agent_pos = np.array(self.np_random.choice(cells))
        self.agent_dir = 0

        cells = []
        for i in range(width-1, 0, -1):
            if (self.grid.get(height-2,i) == cell):
                cells.append((height-2, i))
        
        pt = self.np_random.choice(cells)
        self.put_obj(Goal(), pt[0], pt[1])
        
        # Mark the remaining unvisited cells as walls
        for i in range(0, height):
            for j in range(0, width):
                if (self.grid.get(i,j) is None):
                    self.put_obj(self.obstacle_type(), i, j)
                if (self.grid.get(i,j) == cell):
                    self.grid.set(i, j, None)

        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )
