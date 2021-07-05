import numpy as np

from gym_minigrid.minigrid import MiniGridEnv, Grid, Goal, Lava, Wall


class DynamicMiniGrid(MiniGridEnv):
    """
    DynamicMiniGrid: Mini Grid Environment, that can dynamically change, by altering a single tile
    """

    def __init__(self, size=8, agent_start_pos=(1, 1),agent_start_dir=0, agent_view_size=7):

        # Copied from EmptyEnv Todo: Make this class a child of EmptyEnv?
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        super().__init__( grid_size=size, max_steps=4 * size * size,
                          see_through_walls=False, agent_view_size=agent_view_size)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner # Todo - should this change?
        self.goal_pos = (width-2, height-2)
        self.put_obj(Goal(), *self.goal_pos)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

    def alter(self, prob_array, visibility_check):
        """
        Changes the environment. With

        :param prob_array: numpy.Array. array of probabilties for each type of altering (change start or goal position, add wall/lava)
            prob_array[0]: change start pos; prob_array[1]: change goal position; prob_array[2] add wall at random (allowed) location
            prob_array[3]: add lava at random location
        :param visibility_check: bool. If true, checks whether the agent can see the reward at the start and rejects such a solution
        :return:
        """
    # Todo: Solvability check
    # Todo: add int for multiple repetition

        if np.sum(prob_array) != 1.0:
            raise ValueError('Probabilities do not sum to 1')

        if len(prob_array) != 4:
            raise ValueError('Prob array must be of length 4: start, reward, wall, lava')

        def alter_start_pos():

            def goal_in_view(pos, new_pos):
                self.agent_pos = new_pos
                return_value = self.in_view(*self.goal_pos)
                self.agent_pos = pos

                return return_value

            pos = self.agent_start_pos
            while pos == self.agent_start_pos:
                new_pos = (self.np_random.randint(1, self.height - 1), # 1, -1 to avoid boarders
                           self.np_random.randint(1, self.width - 1))
                if self.grid.get(*new_pos) is not None or new_pos == pos: # check field is empty and agent is not there
                    continue
                if visibility_check and goal_in_view(pos, new_pos):
                    continue
            self.agent_start_pos = new_pos
            self.agent_pos = new_pos
            # self.agent_start_dir = ? todo

        def alter_goal_pos():
            goal_pos = self.goal_pos
            goal = Goal()
            while goal_pos == self.goal_pos:
                new_goal_pos = (self.np_random.randint(1, self.height-1),
                            self.np_random.randint(1, self.width-1))
                if self.grid.get(*new_goal_pos) is not None or new_goal_pos == self.agent_start_pos:
                    continue
                if visibility_check and self.in_view(*new_goal_pos):
                    continue
            self.grid.set(*new_goal_pos, Goal())
            self.grid.set(*goal_pos, None)

        def set_or_remove_obj(obj):

            while True:
                rand_pos = (self.np_random.randint(1, self.height-1),
                            self.np_random.randint(1, self.width-1))

                if rand_pos == self.agent_start_pos or rand_pos == self.goal_pos:
                    continue

                if self.grid.get(*rand_pos) == obj:
                    # remove obj
                    self.grid.set(*rand_pos, None)
                else: # replace even if there is an object of the other type
                    self.grid.set(*rand_pos, obj)
                break


        random_float = self.np_random.uniform()

        if random_float < prob_array[0]:
            alter_start_pos()

        elif random_float < np.sum(prob_array[0:1]):
            alter_goal_pos()

        elif random_float < np.sum(prob_array[0:2]):
            set_or_remove_obj(Wall())

        else:
            set_or_remove_obj(Lava())




# Tests: 1000x alter -