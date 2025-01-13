from __future__ import annotations

import hashlib
import math
from abc import abstractmethod
from typing import Any, Iterable, SupportsFloat, TypeVar

import gymnasium as gym
import numpy as np
import pygame
import pygame.freetype
from gymnasium import spaces
from gymnasium.core import ActType, ObsType

from minigrid.core.actions import Actions
from minigrid.core.constants import (
    COLOR_NAMES,
    COLOR_TO_IDX,
    DIR_TO_VEC,
    OBJECT_TO_IDX,
    TILE_PIXELS,
)
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Point, WorldObj

T = TypeVar("T")


class MiniGridEnv(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        mission_space: MissionSpace,
        grid_size: int | None = None,
        width: int | None = None,
        height: int | None = None,
        max_steps: int = 100,
        see_through_walls: bool = False,
        agent_view_size: int = 7,
        render_mode: str | None = None,
        screen_size: int | None = 640,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
        language="english",
    ):
        # Initialize mission
        self.mission = mission_space.sample()

        # Can't set both grid_size and width/height
        if grid_size:
            assert width is None and height is None
            width = grid_size
            height = grid_size
        assert width is not None and height is not None

        # Action enumeration for this environment
        self.actions = Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        # Number of cells (width and height) in the agent view
        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3
        self.agent_view_size = agent_view_size

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        image_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, 3),
            dtype="uint8",
        )
        self.observation_space = spaces.Dict(
            {
                "image": image_observation_space,
                "direction": spaces.Discrete(4),
                "mission": mission_space,
            }
        )

        # Range of possible rewards
        self.reward_range = (0, 1)

        self.screen_size = screen_size
        self.render_size = None
        self.window = None
        self.clock = None

        # Environment configuration
        self.width = width
        self.height = height

        assert isinstance(
            max_steps, int
        ), f"The argument max_steps must be an integer, got: {type(max_steps)}"
        self.max_steps = max_steps

        self.see_through_walls = see_through_walls

        # Language of the descriptions
        self.language = language

        # Current position and direction of the agent
        self.agent_pos: np.ndarray | tuple[int, int] = None
        self.agent_dir: int = None

        # Current grid and mission and carrying
        self.grid = Grid(width, height)
        self.carrying = None

        # Rendering attributes
        self.render_mode = render_mode
        self.highlight = highlight
        self.tile_size = tile_size
        self.agent_pov = agent_pov

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        # Reinitialize episode-specific variables
        self.agent_pos = (-1, -1)
        self.agent_dir = -1

        # Generate a new random grid at the start of each episode
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert (
            self.agent_pos >= (0, 0)
            if isinstance(self.agent_pos, tuple)
            else all(self.agent_pos >= 0) and self.agent_dir >= 0
        )

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        if self.render_mode == "human":
            self.render()

        # Return first observation
        obs = self.gen_obs()

        # add info Episodic Knowledge to minigrid
        info = self.gen_graph(move_forward=None)

        return obs, info

    def hash(self, size=16):
        """Compute a hash that uniquely identifies the current state of the environment.
        :param size: Size of the hashing
        """
        sample_hash = hashlib.sha256()

        to_encode = [self.grid.encode().tolist(), self.agent_pos, self.agent_dir]
        for item in to_encode:
            sample_hash.update(str(item).encode("utf8"))

        return sample_hash.hexdigest()[:size]

    @property
    def steps_remaining(self):
        return self.max_steps - self.step_count

    def pprint_grid(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """
        if self.agent_pos is None or self.agent_dir is None or self.grid is None:
            raise ValueError(
                "The environment hasn't been `reset` therefore the `agent_pos`, `agent_dir` or `grid` are unknown."
            )

        # Map of object types to short string
        OBJECT_TO_STR = {
            "wall": "W",
            "floor": "F",
            "door": "D",
            "key": "K",
            "ball": "A",
            "box": "B",
            "goal": "G",
            "lava": "V",
        }

        # Map agent's direction to short string
        AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}

        output = ""

        # check if self.agent_pos & self.agent_dir is None
        # should not be after env is reset
        if self.agent_pos is None:
            return super().__str__()

        for j in range(self.grid.height):
            for i in range(self.grid.width):
                if i == self.agent_pos[0] and j == self.agent_pos[1]:
                    output += 2 * AGENT_DIR_TO_STR[self.agent_dir]
                    continue

                tile = self.grid.get(i, j)

                if tile is None:
                    output += "  "
                    continue

                if tile.type == "door":
                    if tile.is_open:
                        output += "__"
                    elif tile.is_locked:
                        output += "L" + tile.color[0].upper()
                    else:
                        output += "D" + tile.color[0].upper()
                    continue

                output += OBJECT_TO_STR[tile.type] + tile.color[0].upper()

            if j < self.grid.height - 1:
                output += "\n"

        return output

    @abstractmethod
    def _gen_grid(self, width, height):
        pass

    def _reward(self) -> float:
        """
        Compute the reward to be given upon success
        """

        return 1 - 0.9 * (self.step_count / self.max_steps)

    def _rand_int(self, low: int, high: int) -> int:
        """
        Generate random integer in [low,high[
        """

        return self.np_random.integers(low, high)

    def _rand_float(self, low: float, high: float) -> float:
        """
        Generate random float in [low,high[
        """

        return self.np_random.uniform(low, high)

    def _rand_bool(self) -> bool:
        """
        Generate random boolean value
        """

        return self.np_random.integers(0, 2) == 0

    def _rand_elem(self, iterable: Iterable[T]) -> T:
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_subset(self, iterable: Iterable[T], num_elems: int) -> list[T]:
        """
        Sample a random subset of distinct elements of a list
        """

        lst = list(iterable)
        assert num_elems <= len(lst)

        out: list[T] = []

        while len(out) < num_elems:
            elem = self._rand_elem(lst)
            lst.remove(elem)
            out.append(elem)

        return out

    def _rand_color(self) -> str:
        """
        Generate a random color name (string)
        """

        return self._rand_elem(COLOR_NAMES)

    def _rand_pos(
        self, x_low: int, x_high: int, y_low: int, y_high: int
    ) -> tuple[int, int]:
        """
        Generate a random (x,y) position tuple
        """

        return (
            self.np_random.integers(x_low, x_high),
            self.np_random.integers(y_low, y_high),
        )

    def place_obj(
        self,
        obj: WorldObj | None,
        top: Point = None,
        size: tuple[int, int] = None,
        reject_fn=None,
        max_tries=math.inf,
    ):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError("rejection sampling failed in place_obj")

            num_tries += 1

            pos = (
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height)),
            )

            # Don't place the object on top of another object
            if self.grid.get(*pos) is not None:
                continue

            # Don't place the object where the agent is
            if np.array_equal(pos, self.agent_pos):
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(pos[0], pos[1], obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def put_obj(self, obj: WorldObj, i: int, j: int):
        """
        Put an object at a specific position in the grid
        """

        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)

    def place_agent(self, top=None, size=None, rand_dir=True, max_tries=math.inf):
        """
        Set the agent's starting point at an empty position in the grid
        """

        self.agent_pos = (-1, -1)
        pos = self.place_obj(None, top, size, max_tries=max_tries)
        self.agent_pos = pos

        if rand_dir:
            self.agent_dir = self._rand_int(0, 4)

        return pos

    @property
    def dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """

        assert (
            self.agent_dir >= 0 and self.agent_dir < 4
        ), f"Invalid agent_dir: {self.agent_dir} is not within range(0, 4)"
        return DIR_TO_VEC[self.agent_dir]

    @property
    def right_vec(self):
        """
        Get the vector pointing to the right of the agent.
        """

        dx, dy = self.dir_vec
        return np.array((-dy, dx))

    @property
    def front_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """

        return self.agent_pos + self.dir_vec

    def get_view_coords(self, i, j):
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        agent's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the agent's view size.
        """

        ax, ay = self.agent_pos
        dx, dy = self.dir_vec
        rx, ry = self.right_vec

        # Compute the absolute coordinates of the top-left view corner
        sz = self.agent_view_size
        hs = self.agent_view_size // 2
        tx = ax + (dx * (sz - 1)) - (rx * hs)
        ty = ay + (dy * (sz - 1)) - (ry * hs)

        lx = i - tx
        ly = j - ty

        # Project the coordinates of the object relative to the top-left
        # corner onto the agent's own coordinate system
        vx = rx * lx + ry * ly
        vy = -(dx * lx + dy * ly)

        return vx, vy

    def get_view_exts(self, agent_view_size=None):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        if agent_view_size is None, use self.agent_view_size
        """

        agent_view_size = agent_view_size or self.agent_view_size

        # Facing right
        if self.agent_dir == 0:
            topX = self.agent_pos[0]
            topY = self.agent_pos[1] - agent_view_size // 2
        # Facing down
        elif self.agent_dir == 1:
            topX = self.agent_pos[0] - agent_view_size // 2
            topY = self.agent_pos[1]
        # Facing left
        elif self.agent_dir == 2:
            topX = self.agent_pos[0] - agent_view_size + 1
            topY = self.agent_pos[1] - agent_view_size // 2
        # Facing up
        elif self.agent_dir == 3:
            topX = self.agent_pos[0] - agent_view_size // 2
            topY = self.agent_pos[1] - agent_view_size + 1
        else:
            assert False, "invalid agent direction"

        botX = topX + agent_view_size
        botY = topY + agent_view_size

        return topX, topY, botX, botY

    def relative_coords(self, x, y):
        """
        Check if a grid position belongs to the agent's field of view, and returns the corresponding coordinates
        """

        vx, vy = self.get_view_coords(x, y)

        if vx < 0 or vy < 0 or vx >= self.agent_view_size or vy >= self.agent_view_size:
            return None

        return vx, vy

    def in_view(self, x, y):
        """
        check if a grid position is visible to the agent
        """

        return self.relative_coords(x, y) is not None

    def agent_sees(self, x, y):
        """
        Check if a non-empty grid position is visible to the agent
        """

        coordinates = self.relative_coords(x, y)
        if coordinates is None:
            return False
        vx, vy = coordinates

        obs = self.gen_obs()

        obs_grid, _ = Grid.decode(obs["image"])
        obs_cell = obs_grid.get(vx, vy)
        world_cell = self.grid.get(x, y)

        assert world_cell is not None

        return obs_cell is not None and obs_cell.type == world_cell.type

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self._reward()
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        # add info Episodic Knowledge to minigrid
        move_forward = None
        if action == self.actions.forward:
            move_forward = False
            if np.all(self.agent_pos == fwd_pos):
                move_forward = True
        info = self.gen_graph(move_forward=move_forward)

        return obs, reward, terminated, truncated, info

    def gen_obs_grid(self, agent_view_size=None):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        if agent_view_size is None, self.agent_view_size is used
        """

        topX, topY, botX, botY = self.get_view_exts(agent_view_size)

        agent_view_size = agent_view_size or self.agent_view_size

        grid = self.grid.slice(topX, topY, agent_view_size, agent_view_size)

        for i in range(self.agent_dir + 1):
            grid = grid.rotate_left()

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(
                agent_pos=(agent_view_size // 2, agent_view_size - 1)
            )
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        agent_pos = grid.width // 2, grid.height - 1
        if self.carrying:
            grid.set(*agent_pos, self.carrying)
        else:
            grid.set(*agent_pos, None)

        return grid, vis_mask

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        obs = {"image": image, "direction": self.agent_dir, "mission": self.mission}

        return obs

    def get_pov_render(self, tile_size):
        """
        Render an agent's POV observation for visualization
        """
        grid, vis_mask = self.gen_obs_grid()

        # Render the whole grid
        img = grid.render(
            tile_size,
            agent_pos=(self.agent_view_size // 2, self.agent_view_size - 1),
            agent_dir=3,
            highlight_mask=vis_mask,
        )

        return img

    def get_full_render(self, highlight, tile_size):
        """
        Render a non-paratial observation for visualization
        """
        # Compute which cells are visible to the agent
        _, vis_mask = self.gen_obs_grid()

        # Compute the world coordinates of the bottom-left corner
        # of the agent's view area
        f_vec = self.dir_vec
        r_vec = self.right_vec
        top_left = (
            self.agent_pos
            + f_vec * (self.agent_view_size - 1)
            - r_vec * (self.agent_view_size // 2)
        )

        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # For each cell in the visibility mask
        for vis_j in range(0, self.agent_view_size):
            for vis_i in range(0, self.agent_view_size):
                # If this cell is not visible, don't highlight it
                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                if abs_i < 0 or abs_i >= self.width:
                    continue
                if abs_j < 0 or abs_j >= self.height:
                    continue

                # Mark this cell to be highlighted
                highlight_mask[abs_i, abs_j] = True

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=highlight_mask if highlight else None,
        )

        return img

    def get_frame(
        self,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
    ):
        """Returns an RGB image corresponding to the whole environment or the agent's point of view.

        Args:

            highlight (bool): If true, the agent's field of view or point of view is highlighted with a lighter gray color.
            tile_size (int): How many pixels will form a tile from the NxM grid.
            agent_pov (bool): If true, the rendered frame will only contain the point of view of the agent.

        Returns:

            frame (np.ndarray): A frame of type numpy.ndarray with shape (x, y, 3) representing RGB values for the x-by-y pixel image.

        """

        if agent_pov:
            return self.get_pov_render(tile_size)
        else:
            return self.get_full_render(highlight, tile_size)

    def render(self):
        img = self.get_frame(self.highlight, self.tile_size, self.agent_pov)

        if self.render_mode == "human":
            img = np.transpose(img, axes=(1, 0, 2))
            if self.render_size is None:
                self.render_size = img.shape[:2]
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.screen_size, self.screen_size)
                )
                pygame.display.set_caption("minigrid")
            if self.clock is None:
                self.clock = pygame.time.Clock()
            surf = pygame.surfarray.make_surface(img)

            # Create background with mission description
            offset = surf.get_size()[0] * 0.1
            # offset = 32 if self.agent_pov else 64
            bg = pygame.Surface(
                (int(surf.get_size()[0] + offset), int(surf.get_size()[1] + offset))
            )
            bg.convert()
            bg.fill((255, 255, 255))
            bg.blit(surf, (offset / 2, 0))

            bg = pygame.transform.smoothscale(bg, (self.screen_size, self.screen_size))

            font_size = 22
            text = self.mission
            font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)
            text_rect = font.get_rect(text, size=font_size)
            text_rect.center = bg.get_rect().center
            text_rect.y = bg.get_height() - font_size * 1.5
            font.render_to(bg, text_rect, text, size=font_size)

            self.window.blit(bg, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return img

    def close(self):
        if self.window:
            pygame.quit()

    def gen_graph(self, move_forward=None):
        grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)
        # (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], state)
        # State, 0: open, 1: closed, 2: locked
        if self.language == "english":
            IDX_TO_STATE = {0: "open", 1: "closed", 2: "locked"}
            IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))
            IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

        elif self.language == "french":
            IDX_TO_STATE = {0: "ouverte", 1: "fermée", 2: "fermée à clef"}
            IDX_TO_COLOR = {
                0: "rouge",
                1: "verte",
                2: "bleue",
                3: "violette",
                4: "jaune",
                5: "grise",
            }
            IDX_TO_OBJECT = {
                0: "non visible",
                1: "vide",
                2: "mur",
                3: "sol",
                4: "porte",
                5: "clef",
                6: "balle",
                7: "boîte",
                8: "but",
                9: "lave",
                10: "agent",
            }

        list_textual_descriptions = []

        if self.carrying is not None:
            # print('carrying')
            if self.language == "english":
                list_textual_descriptions.append(
                    f"You carry a {self.carrying.color} {self.carrying.type}"
                )
            elif self.language == "french":
                list_textual_descriptions.append(
                    "Tu portes une {} {}".format(
                        self.carrying.type, self.carrying.color
                    )
                )

        # print('A agent position i: {}, j: {}'.format(self.agent_pos[0], self.agent_pos[1]))
        agent_pos_vx, agent_pos_vy = self.get_view_coords(
            self.agent_pos[0], self.agent_pos[1]
        )
        # print('B agent position i: {}, j: {}'.format(agent_pos_vx, agent_pos_vy))

        view_field_dictionary = dict()

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j][0] != 0 and image[i][j][0] != 1 and image[i][j][0] != 2:
                    if i not in view_field_dictionary.keys():
                        view_field_dictionary[i] = dict()
                        view_field_dictionary[i][j] = image[i][j]
                    else:
                        view_field_dictionary[i][j] = image[i][j]

        # Find the wall if any
        #  We describe a wall only if there is no objects between the agent and the wall in straight line

        # Find wall in front
        j = agent_pos_vy - 1
        object_seen = False
        while j >= 0 and not object_seen:
            if image[agent_pos_vx][j][0] != 0 and image[agent_pos_vx][j][0] != 1:
                if image[agent_pos_vx][j][0] == 2:
                    if self.language == "english":
                        list_textual_descriptions.append(
                            f"You see a wall {agent_pos_vy - j} step{'s' if agent_pos_vy - j > 1 else ''} forward"
                        )
                    elif self.language == "french":
                        list_textual_descriptions.append(
                            f"Tu vois un mur à {agent_pos_vy - j} pas devant"
                        )
                    object_seen = True
                else:
                    object_seen = True
            j -= 1
        # Find wall left
        i = agent_pos_vx - 1
        object_seen = False
        while i >= 0 and not object_seen:
            if image[i][agent_pos_vy][0] != 0 and image[i][agent_pos_vy][0] != 1:
                if image[i][agent_pos_vy][0] == 2:
                    if self.language == "english":
                        list_textual_descriptions.append(
                            f"You see a wall {agent_pos_vx - i} step{'s' if agent_pos_vx - i > 1 else ''} left"
                        )
                    elif self.language == "french":
                        list_textual_descriptions.append(
                            f"Tu vois un mur à {agent_pos_vx - i} pas à gauche"
                        )
                    object_seen = True
                else:
                    object_seen = True
            i -= 1
        # Find wall right
        i = agent_pos_vx + 1
        object_seen = False
        while i < image.shape[0] and not object_seen:
            if image[i][agent_pos_vy][0] != 0 and image[i][agent_pos_vy][0] != 1:
                if image[i][agent_pos_vy][0] == 2:
                    if self.language == "english":
                        list_textual_descriptions.append(
                            f"You see a wall {i - agent_pos_vx} step{'s' if i - agent_pos_vx > 1 else ''} right"
                        )
                    elif self.language == "french":
                        list_textual_descriptions.append(
                            f"Tu vois un mur à {i - agent_pos_vx} pas à droite"
                        )
                    object_seen = True
                else:
                    object_seen = True
            i += 1

        # returns the position of seen objects relative to you
        for i in view_field_dictionary.keys():
            for j in view_field_dictionary[i].keys():
                if i != agent_pos_vx or j != agent_pos_vy:
                    object = view_field_dictionary[i][j]
                    relative_position = dict()

                    if i - agent_pos_vx > 0:
                        if self.language == "english":
                            relative_position["x_axis"] = ("right", i - agent_pos_vx)
                        elif self.language == "french":
                            relative_position["x_axis"] = ("à droite", i - agent_pos_vx)
                    elif i - agent_pos_vx == 0:
                        if self.language == "english":
                            relative_position["x_axis"] = ("face", 0)
                        elif self.language == "french":
                            relative_position["x_axis"] = ("en face", 0)
                    else:
                        if self.language == "english":
                            relative_position["x_axis"] = ("left", agent_pos_vx - i)
                        elif self.language == "french":
                            relative_position["x_axis"] = ("à gauche", agent_pos_vx - i)
                    if agent_pos_vy - j > 0:
                        if self.language == "english":
                            relative_position["y_axis"] = ("forward", agent_pos_vy - j)
                        elif self.language == "french":
                            relative_position["y_axis"] = ("devant", agent_pos_vy - j)
                    elif agent_pos_vy - j == 0:
                        if self.language == "english":
                            relative_position["y_axis"] = ("forward", 0)
                        elif self.language == "french":
                            relative_position["y_axis"] = ("devant", 0)

                    distances = []
                    if relative_position["x_axis"][0] in ["face", "en face"]:
                        distances.append(
                            (
                                relative_position["y_axis"][1],
                                relative_position["y_axis"][0],
                            )
                        )
                    elif relative_position["y_axis"][1] == 0:
                        distances.append(
                            (
                                relative_position["x_axis"][1],
                                relative_position["x_axis"][0],
                            )
                        )
                    else:
                        distances.append(
                            (
                                relative_position["x_axis"][1],
                                relative_position["x_axis"][0],
                            )
                        )
                        distances.append(
                            (
                                relative_position["y_axis"][1],
                                relative_position["y_axis"][0],
                            )
                        )

                    description = ""
                    if object[0] != 4:  # if it is not a door
                        if self.language == "english":
                            description = f"You see a {IDX_TO_COLOR[object[1]]} {IDX_TO_OBJECT[object[0]]} "
                        elif self.language == "french":
                            description = f"Tu vois une {IDX_TO_OBJECT[object[0]]} {IDX_TO_COLOR[object[1]]} "

                    else:
                        if IDX_TO_STATE[object[2]] != 0:  # if it is not open
                            if self.language == "english":
                                description = f"You see a {IDX_TO_STATE[object[2]]} {IDX_TO_COLOR[object[1]]} {IDX_TO_OBJECT[object[0]]} "
                            elif self.language == "french":
                                description = f"Tu vois une {IDX_TO_OBJECT[object[0]]} {IDX_TO_COLOR[object[1]]} {IDX_TO_STATE[object[2]]} "

                        else:
                            if self.language == "english":
                                description = f"You see an {IDX_TO_STATE[object[2]]} {IDX_TO_COLOR[object[1]]} {IDX_TO_OBJECT[object[0]]} "
                            elif self.language == "french":
                                description = f"Tu vois une {IDX_TO_OBJECT[object[0]]} {IDX_TO_COLOR[object[1]]} {IDX_TO_STATE[object[2]]} "

                    for _i, _distance in enumerate(distances):
                        if _i > 0:
                            if self.language == "english":
                                description += " and "
                            elif self.language == "french":
                                description += " et "

                        if self.language == "english":
                            description += f"{_distance[0]} step{'s' if _distance[0] > 1 else ''} {_distance[1]}"
                        elif self.language == "french":
                            description += f"{_distance[0]} pas {_distance[1]}"

                    list_textual_descriptions.append(description)

        return {"descriptions": list_textual_descriptions}
