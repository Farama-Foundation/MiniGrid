"""
Copied and adapted from https://github.com/flowersteam/Grounding_LLMs_with_online_RL
"""

from __future__ import annotations

from minigrid.envs.babyai.core.levelgen import (
    GoToInstr,
    LevelGen,
    PickupInstr,
    PutNextInstr,
)
from minigrid.envs.babyai.core.verifier import (
    AfterInstr,
    BeforeInstr,
    ObjDesc,
    OpenInstr,
)


class Level_MixedTrainLocal(LevelGen):
    """
    Union of all instructions from PutNext, Open, Goto and PickUp.
    The agent does not need to move objects around.
    When the task is Open there are 2 rooms and the door in between is locked.
    For the other instructions there is only one room.
    Sequence of action are possible.

    In order to test generalisation we do not give to the agent the instructions containing:
    - yellow box
    - red door/key
    - green ball
    - grey door
    - seq is restricted to pick up A then/before go to B (for memory issue our agent only used the past 3 observations)

    At test time we release the 3 first previous constraints, and we add to seq
    pick up A then/before pick up B

    Competencies: Unlock, GoTo, PickUp, PutNext, Seq
    """

    def __init__(
        self,
        room_size=8,
        num_rows=1,
        num_cols=1,
        num_dists=8,
        instr_kinds=["action", "seq1"],
        locations=False,
        unblocking=False,
        implicit_unlock=False,
        **kwargs,
    ):
        action = self._rand_elem(
            ["goto", "pickup", "open", "putnext", "pick up seq go to"]
        )
        if action == "open":
            num_cols = 2
            num_rows = 1
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            action_kinds=[action],
            instr_kinds=instr_kinds,
            locations=locations,
            unblocking=unblocking,
            implicit_unlock=implicit_unlock,
            **kwargs,
        )

    # ['goto', 'pickup', 'open', 'putnext', 'pick up seq go to'],
    def gen_mission(self):
        action = self._rand_elem(self.action_kinds)
        mission_accepted = False
        all_objects_reachable = False
        if action == "open":
            while not mission_accepted or not all_objects_reachable:
                self._regen_grid()
                color_door = self._rand_elem(
                    ["yellow", "green", "blue", "purple"]
                )  # red and grey excluded
                self.add_locked_room(color_door)
                self.connect_all()

                for j in range(self.num_rows):
                    for i in range(self.num_cols):
                        if self.get_room(i, j) is not self.locked_room:
                            self.add_distractors(
                                i, j, num_distractors=self.num_dists, all_unique=False
                            )

                # The agent must be placed after all the object to respect constraints
                while True:
                    self.place_agent()
                    start_room = self.room_from_pos(*self.agent_pos)
                    # Ensure that we are not placing the agent in the locked room
                    if start_room is self.locked_room:
                        continue
                    break

                all_objects_reachable = self.check_objs_reachable(raise_exc=False)

                color_in_instr = self._rand_elem([None, color_door])

                desc = ObjDesc("door", color_in_instr)
                self.instrs = OpenInstr(desc)

                mission_accepted = not (self.exclude_substrings())

                """if color_in_instr is None and mission_accepted and all_objects_reachable:
                    print(color_door)"""

        elif action == "goto":
            self.num_cols = 1
            self.num_rows = 1
            while not mission_accepted or not all_objects_reachable:
                self._regen_grid()
                self.place_agent()
                objs = self.add_distractors(
                    num_distractors=self.num_dists + 1, all_unique=False
                )
                all_objects_reachable = self.check_objs_reachable(raise_exc=False)
                obj = self._rand_elem(objs)
                self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))

                mission_accepted = not (self.exclude_substrings())

        elif action == "pickup":
            self.num_cols = 1
            self.num_rows = 1
            while not mission_accepted or not all_objects_reachable:
                self._regen_grid()
                self.place_agent()
                objs = self.add_distractors(
                    num_distractors=self.num_dists + 1, all_unique=False
                )
                all_objects_reachable = self.check_objs_reachable(raise_exc=False)
                obj = self._rand_elem(objs)
                while str(obj.type) == "door":
                    obj = self._rand_elem(objs)
                self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))

                mission_accepted = not (self.exclude_substrings())

        elif action == "putnext":
            self.num_cols = 1
            self.num_rows = 1
            while not mission_accepted or not all_objects_reachable:
                self._regen_grid()
                self.place_agent()
                objs = self.add_distractors(
                    num_distractors=self.num_dists + 2, all_unique=False
                )
                all_objects_reachable = self.check_objs_reachable(raise_exc=False)
                obj_1 = self._rand_elem(objs)
                while str(obj_1.type) == "door":
                    obj_1 = self._rand_elem(objs)
                desc1 = ObjDesc(obj_1.type, obj_1.color)
                obj_2 = self._rand_elem(objs)
                if obj_1.type == obj_2.type and obj_1.color == obj_2.color:
                    obj1s, poss = desc1.find_matching_objs(self)
                    if len(obj1s) < 2:
                        # if obj_1 is the only object with this description obj_2 has to be different
                        while obj_1.type == obj_2.type and obj_1.color == obj_2.color:
                            obj_2 = self._rand_elem(objs)
                desc2 = ObjDesc(obj_2.type, obj_2.color)
                self.instrs = PutNextInstr(desc1, desc2)

                mission_accepted = not (self.exclude_substrings())

        elif action == "pick up seq go to":
            self.num_cols = 1
            self.num_rows = 1
            while not mission_accepted or not all_objects_reachable:
                self._regen_grid()
                self.place_agent()
                objs = self.add_distractors(
                    num_distractors=self.num_dists + 2, all_unique=False
                )
                all_objects_reachable = self.check_objs_reachable(raise_exc=False)
                obj_a = self._rand_elem(objs)
                while str(obj_a.type) == "door":
                    obj_a = self._rand_elem(objs)
                instr_a = PickupInstr(ObjDesc(obj_a.type, obj_a.color))
                obj_b = self._rand_elem(objs)
                if obj_a.type == obj_b.type and obj_a.color == obj_b.color:
                    desc = ObjDesc(obj_a.type, obj_a.color)
                    objas, poss = desc.find_matching_objs(self)
                    if len(objas) < 2:
                        # if obj_a is the only object with this description obj_b has to be different
                        while obj_a.type == obj_b.type and obj_a.color == obj_b.color:
                            obj_b = self._rand_elem(objs)
                instr_b = GoToInstr(ObjDesc(obj_b.type, obj_b.color))

                type_instr = self._rand_elem(["Before", "After"])

                if type_instr == "Before":
                    self.instrs = BeforeInstr(instr_a, instr_b)
                else:
                    self.instrs = AfterInstr(instr_b, instr_a)

                mission_accepted = not (self.exclude_substrings())

    def exclude_substrings(self):
        # True if contains excluded substring
        list_exclude_combinaison = [
            "yellow box",
            "red key",
            "red door",
            "green ball",
            "grey door",
        ]

        for sub_str in list_exclude_combinaison:
            str = self.instrs.surface(self)
            if sub_str in self.instrs.surface(self):
                return True
        return False

    def _regen_grid(self):
        # Create the grid
        self.grid.grid = [None] * self.width * self.height

        # For each row of rooms
        for j in range(0, self.num_rows):
            row = []

            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.get_room(i, j)
                # suppress doors and objects
                room.doors = [None] * 4
                room.door_pos = [None] * 4
                room.neighbors = [None] * 4
                room.locked = False
                room.objs = []
                row.append(room)

                # Generate the walls for this room
                self.grid.wall_rect(*room.top, *room.size)

            self.room_grid.append(row)

        # For each row of rooms
        for j in range(0, self.num_rows):
            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.room_grid[j][i]

                x_l, y_l = (room.top[0] + 1, room.top[1] + 1)
                x_m, y_m = (
                    room.top[0] + room.size[0] - 1,
                    room.top[1] + room.size[1] - 1,
                )

                # Door positions, order is right, down, left, up
                if i < self.num_cols - 1:
                    room.neighbors[0] = self.room_grid[j][i + 1]
                    room.door_pos[0] = (x_m, self._rand_int(y_l, y_m))
                if j < self.num_rows - 1:
                    room.neighbors[1] = self.room_grid[j + 1][i]
                    room.door_pos[1] = (self._rand_int(x_l, x_m), y_m)
                if i > 0:
                    room.neighbors[2] = self.room_grid[j][i - 1]
                    room.door_pos[2] = room.neighbors[2].door_pos[0]
                if j > 0:
                    room.neighbors[3] = self.room_grid[j - 1][i]
                    room.door_pos[3] = room.neighbors[3].door_pos[1]

        # The agent starts in the middle, facing right
        self.agent_pos = (
            (self.num_cols // 2) * (self.room_size - 1) + (self.room_size // 2),
            (self.num_rows // 2) * (self.room_size - 1) + (self.room_size // 2),
        )
        self.agent_dir = 0


class Level_MixedTestLocal(LevelGen):
    """
    Union of all instructions from PutNext, Open, Goto and PickUp.
    The agent does not need to move objects around.
    When the task is Open there are 2 rooms and the door in between is locked.
    For the other instructions there is only one room.
    Sequence of action are possible.

    In order to test generalisation we only give to the agent the instructions containing:
    - yellow box
    - red door/key
    - green ball
    - grey door
    - seq is restricted to pick up A then/before go to B  with A and B among the previous adj-noun pairs
    (for memory issue our agent only used the past 3 observations)

    Competencies: Unlock, GoTo, PickUp, PutNext, Seq
    """

    def __init__(
        self,
        room_size=8,
        num_rows=1,
        num_cols=1,
        num_dists=8,
        instr_kinds=["action", "seq1"],
        locations=False,
        unblocking=False,
        implicit_unlock=False,
        **kwargs,
    ):
        action = self._rand_elem(
            ["goto", "pickup", "open", "putnext", "pick up seq go to"]
        )
        if action == "open":
            num_cols = 2
            num_rows = 1
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            action_kinds=[action],
            instr_kinds=instr_kinds,
            locations=locations,
            unblocking=unblocking,
            implicit_unlock=implicit_unlock,
            **kwargs,
        )

    def gen_mission(self):
        action = self._rand_elem(self.action_kinds)
        mission_accepted = False
        all_objects_reachable = False
        if action == "open":
            while not mission_accepted or not all_objects_reachable:
                self._regen_grid()
                color_door = self._rand_elem(
                    ["red", "grey"]
                )  # only red and grey doors at test time
                self.add_locked_room(color_door)
                self.connect_all()

                for j in range(self.num_rows):
                    for i in range(self.num_cols):
                        if self.get_room(i, j) is not self.locked_room:
                            self.add_distractors(
                                i, j, num_distractors=self.num_dists, all_unique=False
                            )

                # The agent must be placed after all the object to respect constraints
                while True:
                    self.place_agent()
                    start_room = self.room_from_pos(*self.agent_pos)
                    # Ensure that we are not placing the agent in the locked room
                    if start_room is self.locked_room:
                        continue
                    break

                all_objects_reachable = self.check_objs_reachable(raise_exc=False)

                desc = ObjDesc("door", color_door)
                self.instrs = OpenInstr(desc)

                mission_accepted = not (self.exclude_substrings())

        elif action == "goto":
            self.num_cols = 1
            self.num_rows = 1
            while not mission_accepted or not all_objects_reachable:
                self._regen_grid()
                self.place_agent()
                objs = self.add_distractors(
                    num_distractors=self.num_dists + 1, all_unique=False
                )
                all_objects_reachable = self.check_objs_reachable(raise_exc=False)
                obj = self._rand_elem(objs)
                self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))

                mission_accepted = not (self.exclude_substrings())

        elif action == "pickup":
            self.num_cols = 1
            self.num_rows = 1
            while not mission_accepted or not all_objects_reachable:
                self._regen_grid()
                self.place_agent()
                objs = self.add_distractors(
                    num_distractors=self.num_dists + 1, all_unique=False
                )
                all_objects_reachable = self.check_objs_reachable(raise_exc=False)
                obj = self._rand_elem(objs)
                while str(obj.type) == "door":
                    obj = self._rand_elem(objs)
                self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))

                mission_accepted = not (self.exclude_substrings())

        elif action == "putnext":
            self.num_cols = 1
            self.num_rows = 1
            while not mission_accepted or not all_objects_reachable:
                self._regen_grid()
                self.place_agent()
                objs = self.add_distractors(
                    num_distractors=self.num_dists + 2, all_unique=False
                )
                all_objects_reachable = self.check_objs_reachable(raise_exc=False)
                obj_1 = self._rand_elem(objs)
                while str(obj_1.type) == "door":
                    obj_1 = self._rand_elem(objs)
                desc1 = ObjDesc(obj_1.type, obj_1.color)
                obj_2 = self._rand_elem(objs)
                if obj_1.type == obj_2.type and obj_1.color == obj_2.color:
                    obj1s, poss = desc1.find_matching_objs(self)
                    if len(obj1s) < 2:
                        # if obj_1 is the only object with this description obj_2 has to be different
                        while obj_1.type == obj_2.type and obj_1.color == obj_2.color:
                            obj_2 = self._rand_elem(objs)
                desc2 = ObjDesc(obj_2.type, obj_2.color)
                self.instrs = PutNextInstr(desc1, desc2)

                mission_accepted = not (self.exclude_substrings())

        elif action == "pick up seq go to":
            self.num_cols = 1
            self.num_rows = 1
            while not mission_accepted or not all_objects_reachable:
                self._regen_grid()
                self.place_agent()
                objs = self.add_distractors(
                    num_distractors=self.num_dists + 2, all_unique=False
                )
                all_objects_reachable = self.check_objs_reachable(raise_exc=False)
                obj_a = self._rand_elem(objs)
                while str(obj_a.type) == "door":
                    obj_a = self._rand_elem(objs)
                instr_a = PickupInstr(ObjDesc(obj_a.type, obj_a.color))
                obj_b = self._rand_elem(objs)
                if obj_a.type == obj_b.type and obj_a.color == obj_b.color:
                    desc = ObjDesc(obj_a.type, obj_a.color)
                    objas, poss = desc.find_matching_objs(self)
                    if len(objas) < 2:
                        # if obj_a is the only object with this description obj_b has to be different
                        while obj_a.type == obj_b.type and obj_a.color == obj_b.color:
                            obj_b = self._rand_elem(objs)
                instr_b = GoToInstr(ObjDesc(obj_b.type, obj_b.color))

                type_instr = self._rand_elem(["Before", "After"])

                if type_instr == "Before":
                    self.instrs = BeforeInstr(instr_a, instr_b)
                else:
                    self.instrs = AfterInstr(instr_b, instr_a)

                mission_accepted = not (self.exclude_substrings())

    def exclude_substrings(self):
        # True if contains excluded substring
        list_exclude_combinaison = [
            "yellow key",
            "yellow ball",
            "yellow door",
            "red box",
            "red ball",
            "green box",
            "green key",
            "green door",
            "grey box",
            "grey key",
            "grey ball",
            "blue box",
            "blue key",
            "blue ball",
            "blue door",
            "purple box",
            "purple key",
            "purple ball",
            "purple door",
        ]

        for sub_str in list_exclude_combinaison:
            if sub_str in self.instrs.surface(self):
                return True
        return False

    def _regen_grid(self):
        # Create the grid
        self.grid.grid = [None] * self.width * self.height

        # For each row of rooms
        for j in range(0, self.num_rows):
            row = []

            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.get_room(i, j)
                # suppress doors and objects
                room.doors = [None] * 4
                room.door_pos = [None] * 4
                room.neighbors = [None] * 4
                room.locked = False
                room.objs = []
                row.append(room)

                # Generate the walls for this room
                self.grid.wall_rect(*room.top, *room.size)

            self.room_grid.append(row)

        # For each row of rooms
        for j in range(0, self.num_rows):
            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.room_grid[j][i]

                x_l, y_l = (room.top[0] + 1, room.top[1] + 1)
                x_m, y_m = (
                    room.top[0] + room.size[0] - 1,
                    room.top[1] + room.size[1] - 1,
                )

                # Door positions, order is right, down, left, up
                if i < self.num_cols - 1:
                    room.neighbors[0] = self.room_grid[j][i + 1]
                    room.door_pos[0] = (x_m, self._rand_int(y_l, y_m))
                if j < self.num_rows - 1:
                    room.neighbors[1] = self.room_grid[j + 1][i]
                    room.door_pos[1] = (self._rand_int(x_l, x_m), y_m)
                if i > 0:
                    room.neighbors[2] = self.room_grid[j][i - 1]
                    room.door_pos[2] = room.neighbors[2].door_pos[0]
                if j > 0:
                    room.neighbors[3] = self.room_grid[j - 1][i]
                    room.door_pos[3] = room.neighbors[3].door_pos[1]

        # The agent starts in the middle, facing right
        self.agent_pos = (
            (self.num_cols // 2) * (self.room_size - 1) + (self.room_size // 2),
            (self.num_rows // 2) * (self.room_size - 1) + (self.room_size // 2),
        )
        self.agent_dir = 0


class Level_MixedTrainLocalFrench(LevelGen):
    """
    Same as MixedTrainLocal but in French
    """

    # TODO pas encore fini

    def __init__(
        self,
        room_size=8,
        num_rows=1,
        num_cols=1,
        num_dists=8,
        language="french",
        instr_kinds=["action", "seq1"],
        locations=False,
        unblocking=False,
        implicit_unlock=False,
        **kwargs,
    ):
        action = self._rand_elem(
            ["goto", "pickup", "open", "putnext", "pick up seq go to"]
        )
        if action == "open":
            num_cols = 2
            num_rows = 1
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            action_kinds=[action],
            language=language,
            instr_kinds=instr_kinds,
            locations=locations,
            unblocking=unblocking,
            implicit_unlock=implicit_unlock,
            **kwargs,
        )

    # ['goto', 'pickup', 'open', 'putnext', 'pick up seq go to'],
    def gen_mission(self):
        action = self._rand_elem(self.action_kinds)
        mission_accepted = False
        all_objects_reachable = False
        if action == "open":
            while not mission_accepted or not all_objects_reachable:
                self._regen_grid()
                color_door = self._rand_elem(
                    ["jaune", "verte", "bleue", "violette"]
                )  # red and grey excluded
                self.add_locked_room(color_door)
                self.connect_all()

                for j in range(self.num_rows):
                    for i in range(self.num_cols):
                        if self.get_room(i, j) is not self.locked_room:
                            self.add_distractors(
                                i, j, num_distractors=self.num_dists, all_unique=False
                            )

                # The agent must be placed after all the object to respect constraints
                while True:
                    self.place_agent()
                    start_room = self.room_from_pos(*self.agent_pos)
                    # Ensure that we are not placing the agent in the locked room
                    if start_room is self.locked_room:
                        continue
                    break

                all_objects_reachable = self.check_objs_reachable(raise_exc=False)

                color_in_instr = self._rand_elem([None, color_door])

                desc = ObjDesc("door", color_in_instr)
                self.instrs = OpenInstr(desc)

                mission_accepted = not (self.exclude_substrings())

                """if color_in_instr is None and mission_accepted and all_objects_reachable:
                    print(color_door)"""

        elif action == "goto":
            self.num_cols = 1
            self.num_rows = 1
            while not mission_accepted or not all_objects_reachable:
                self._regen_grid()
                self.place_agent()
                objs = self.add_distractors(
                    num_distractors=self.num_dists + 1, all_unique=False
                )
                all_objects_reachable = self.check_objs_reachable(raise_exc=False)
                obj = self._rand_elem(objs)
                self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))

                mission_accepted = not (self.exclude_substrings())

        elif action == "pickup":
            self.num_cols = 1
            self.num_rows = 1
            while not mission_accepted or not all_objects_reachable:
                self._regen_grid()
                self.place_agent()
                objs = self.add_distractors(
                    num_distractors=self.num_dists + 1, all_unique=False
                )
                all_objects_reachable = self.check_objs_reachable(raise_exc=False)
                obj = self._rand_elem(objs)
                while str(obj.type) == "door":
                    obj = self._rand_elem(objs)
                self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))

                mission_accepted = not (self.exclude_substrings())

        elif action == "putnext":
            self.num_cols = 1
            self.num_rows = 1
            while not mission_accepted or not all_objects_reachable:
                self._regen_grid()
                self.place_agent()
                objs = self.add_distractors(
                    num_distractors=self.num_dists + 2, all_unique=False
                )
                all_objects_reachable = self.check_objs_reachable(raise_exc=False)
                obj_1 = self._rand_elem(objs)
                while str(obj_1.type) == "door":
                    obj_1 = self._rand_elem(objs)
                desc1 = ObjDesc(obj_1.type, obj_1.color)
                obj_2 = self._rand_elem(objs)
                if obj_1.type == obj_2.type and obj_1.color == obj_2.color:
                    obj1s, poss = desc1.find_matching_objs(self)
                    if len(obj1s) < 2:
                        # if obj_1 is the only object with this description obj_2 has to be different
                        while obj_1.type == obj_2.type and obj_1.color == obj_2.color:
                            obj_2 = self._rand_elem(objs)
                desc2 = ObjDesc(obj_2.type, obj_2.color)
                self.instrs = PutNextInstr(desc1, desc2)

                mission_accepted = not (self.exclude_substrings())

        elif action == "pick up seq go to":
            self.num_cols = 1
            self.num_rows = 1
            while not mission_accepted or not all_objects_reachable:
                self._regen_grid()
                self.place_agent()
                objs = self.add_distractors(
                    num_distractors=self.num_dists + 2, all_unique=False
                )
                all_objects_reachable = self.check_objs_reachable(raise_exc=False)
                obj_a = self._rand_elem(objs)
                while str(obj_a.type) == "door":
                    obj_a = self._rand_elem(objs)
                instr_a = PickupInstr(ObjDesc(obj_a.type, obj_a.color))
                obj_b = self._rand_elem(objs)
                if obj_a.type == obj_b.type and obj_a.color == obj_b.color:
                    desc = ObjDesc(obj_a.type, obj_a.color)
                    objas, poss = desc.find_matching_objs(self)
                    if len(objas) < 2:
                        # if obj_a is the only object with this description obj_b has to be different
                        while obj_a.type == obj_b.type and obj_a.color == obj_b.color:
                            obj_b = self._rand_elem(objs)
                instr_b = GoToInstr(ObjDesc(obj_b.type, obj_b.color))

                type_instr = self._rand_elem(["Before", "After"])

                if type_instr == "Before":
                    self.instrs = BeforeInstr(instr_a, instr_b)
                else:
                    self.instrs = AfterInstr(instr_b, instr_a)

                mission_accepted = not (self.exclude_substrings())

    def exclude_substrings(self):
        # True if contains excluded substring
        list_exclude_combinaison = [
            "boîte jaune",
            "clef rouge",
            "porte rouge",
            "balle verte",
            "porte grise",
        ]

        for sub_str in list_exclude_combinaison:
            str = self.instrs.surface(self)
            if sub_str in self.instrs.surface(self):
                return True
        return False

    def _regen_grid(self):
        # Create the grid
        self.grid.grid = [None] * self.width * self.height

        # For each row of rooms
        for j in range(0, self.num_rows):
            row = []

            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.get_room(i, j)
                # suppress doors and objects
                room.doors = [None] * 4
                room.door_pos = [None] * 4
                room.neighbors = [None] * 4
                room.locked = False
                room.objs = []
                row.append(room)

                # Generate the walls for this room
                self.grid.wall_rect(*room.top, *room.size)

            self.room_grid.append(row)

        # For each row of rooms
        for j in range(0, self.num_rows):
            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.room_grid[j][i]

                x_l, y_l = (room.top[0] + 1, room.top[1] + 1)
                x_m, y_m = (
                    room.top[0] + room.size[0] - 1,
                    room.top[1] + room.size[1] - 1,
                )

                # Door positions, order is right, down, left, up
                if i < self.num_cols - 1:
                    room.neighbors[0] = self.room_grid[j][i + 1]
                    room.door_pos[0] = (x_m, self._rand_int(y_l, y_m))
                if j < self.num_rows - 1:
                    room.neighbors[1] = self.room_grid[j + 1][i]
                    room.door_pos[1] = (self._rand_int(x_l, x_m), y_m)
                if i > 0:
                    room.neighbors[2] = self.room_grid[j][i - 1]
                    room.door_pos[2] = room.neighbors[2].door_pos[0]
                if j > 0:
                    room.neighbors[3] = self.room_grid[j - 1][i]
                    room.door_pos[3] = room.neighbors[3].door_pos[1]

        # The agent starts in the middle, facing right
        self.agent_pos = (
            (self.num_cols // 2) * (self.room_size - 1) + (self.room_size // 2),
            (self.num_rows // 2) * (self.room_size - 1) + (self.room_size // 2),
        )
        self.agent_dir = 0


class Level_PickUpSeqGoToLocal(LevelGen):
    """
    In order to test generalisation we only give to the agent the instruction:
    seq restricted to pick up A then/before go to B  with A and B without the following adj-noun pairs:
    - yellow box
    - red door/key
    - green ball
    - grey door
    (for memory issue our agent only used the past 3 observations)

    Competencies: Seq never seen in MixedTrainLocal
    """

    def __init__(
        self,
        room_size=8,
        num_rows=1,
        num_cols=1,
        num_dists=8,
        instr_kinds=["seq1"],
        locations=False,
        unblocking=False,
        implicit_unlock=False,
        **kwargs,
    ):
        action = "pick up seq pick up "

        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            action_kinds=[action],
            instr_kinds=instr_kinds,
            locations=locations,
            unblocking=unblocking,
            implicit_unlock=implicit_unlock,
            **kwargs,
        )

    def gen_mission(self):
        mission_accepted = False
        all_objects_reachable = False

        while not mission_accepted or not all_objects_reachable:
            self._regen_grid()
            self.place_agent()
            objs = self.add_distractors(
                num_distractors=self.num_dists + 2, all_unique=False
            )
            all_objects_reachable = self.check_objs_reachable(raise_exc=False)
            obj_a = self._rand_elem(objs)
            while str(obj_a.type) == "door":
                obj_a = self._rand_elem(objs)
            instr_a = PickupInstr(ObjDesc(obj_a.type, obj_a.color))
            obj_b = self._rand_elem(objs)
            if obj_a.type == obj_b.type and obj_a.color == obj_b.color:
                desc = ObjDesc(obj_a.type, obj_a.color)
                objas, poss = desc.find_matching_objs(self)
                if len(objas) < 2:
                    # if obj_a is the only object with this description obj_b has to be different
                    while obj_a.type == obj_b.type and obj_a.color == obj_b.color:
                        obj_b = self._rand_elem(objs)
            instr_b = GoToInstr(ObjDesc(obj_b.type, obj_b.color))

            type_instr = self._rand_elem(["Before", "After"])

            if type_instr == "Before":
                self.instrs = BeforeInstr(instr_a, instr_b)
            else:
                self.instrs = AfterInstr(instr_b, instr_a)

            mission_accepted = not (self.exclude_substrings())

    def exclude_substrings(self):
        # True if contains excluded substring
        list_exclude_combinaison = [
            "yellow box",
            "red key",
            "red door",
            "green ball",
            "grey door",
        ]

        for sub_str in list_exclude_combinaison:
            if sub_str in self.instrs.surface(self):
                return True
        return False

    def _regen_grid(self):
        # Create the grid
        self.grid.grid = [None] * self.width * self.height

        # For each row of rooms
        for j in range(0, self.num_rows):
            row = []

            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.get_room(i, j)
                # suppress doors and objects
                room.doors = [None] * 4
                room.door_pos = [None] * 4
                room.neighbors = [None] * 4
                room.locked = False
                room.objs = []
                row.append(room)

                # Generate the walls for this room
                self.grid.wall_rect(*room.top, *room.size)

            self.room_grid.append(row)

        # For each row of rooms
        for j in range(0, self.num_rows):
            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.room_grid[j][i]

                x_l, y_l = (room.top[0] + 1, room.top[1] + 1)
                x_m, y_m = (
                    room.top[0] + room.size[0] - 1,
                    room.top[1] + room.size[1] - 1,
                )

                # Door positions, order is right, down, left, up
                if i < self.num_cols - 1:
                    room.neighbors[0] = self.room_grid[j][i + 1]
                    room.door_pos[0] = (x_m, self._rand_int(y_l, y_m))
                if j < self.num_rows - 1:
                    room.neighbors[1] = self.room_grid[j + 1][i]
                    room.door_pos[1] = (self._rand_int(x_l, x_m), y_m)
                if i > 0:
                    room.neighbors[2] = self.room_grid[j][i - 1]
                    room.door_pos[2] = room.neighbors[2].door_pos[0]
                if j > 0:
                    room.neighbors[3] = self.room_grid[j - 1][i]
                    room.door_pos[3] = room.neighbors[3].door_pos[1]

        # The agent starts in the middle, facing right
        self.agent_pos = (
            (self.num_cols // 2) * (self.room_size - 1) + (self.room_size // 2),
            (self.num_rows // 2) * (self.room_size - 1) + (self.room_size // 2),
        )
        self.agent_dir = 0


class Level_PickUpThenGoToLocal(LevelGen):
    """
    In order to test generalisation we only give to the agent the instruction:
    seq restricted to pick up A then go to B  with A and B without the following adj-noun pairs:
    - yellow box
    - red door/key
    - green ball
    - grey door
    (for memory issue our agent only used the past 3 observations)

    Competencies: Seq never seen in MixedTrainLocal
    """

    def __init__(
        self,
        room_size=8,
        num_rows=1,
        num_cols=1,
        num_dists=8,
        instr_kinds=["seq1"],
        locations=False,
        unblocking=False,
        implicit_unlock=False,
        **kwargs,
    ):
        action = "pick up seq pick up "

        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            action_kinds=[action],
            instr_kinds=instr_kinds,
            locations=locations,
            unblocking=unblocking,
            implicit_unlock=implicit_unlock,
            **kwargs,
        )

    def gen_mission(self):
        mission_accepted = False
        all_objects_reachable = False

        while not mission_accepted or not all_objects_reachable:
            self._regen_grid()
            self.place_agent()
            objs = self.add_distractors(
                num_distractors=self.num_dists + 2, all_unique=False
            )
            all_objects_reachable = self.check_objs_reachable(raise_exc=False)
            obj_a = self._rand_elem(objs)
            while str(obj_a.type) == "door":
                obj_a = self._rand_elem(objs)
            instr_a = PickupInstr(ObjDesc(obj_a.type, obj_a.color))
            obj_b = self._rand_elem(objs)
            if obj_a.type == obj_b.type and obj_a.color == obj_b.color:
                desc = ObjDesc(obj_a.type, obj_a.color)
                objas, poss = desc.find_matching_objs(self)
                if len(objas) < 2:
                    # if obj_a is the only object with this description obj_b has to be different
                    while obj_a.type == obj_b.type and obj_a.color == obj_b.color:
                        obj_b = self._rand_elem(objs)
            instr_b = GoToInstr(ObjDesc(obj_b.type, obj_b.color))

            self.instrs = BeforeInstr(instr_a, instr_b)

            mission_accepted = not (self.exclude_substrings())

    def exclude_substrings(self):
        # True if contains excluded substring
        list_exclude_combinaison = [
            "yellow box",
            "red key",
            "red door",
            "green ball",
            "grey door",
        ]

        for sub_str in list_exclude_combinaison:
            if sub_str in self.instrs.surface(self):
                return True
        return False

    def _regen_grid(self):
        # Create the grid
        self.grid.grid = [None] * self.width * self.height

        # For each row of rooms
        for j in range(0, self.num_rows):
            row = []

            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.get_room(i, j)
                # suppress doors and objects
                room.doors = [None] * 4
                room.door_pos = [None] * 4
                room.neighbors = [None] * 4
                room.locked = False
                room.objs = []
                row.append(room)

                # Generate the walls for this room
                self.grid.wall_rect(*room.top, *room.size)

            self.room_grid.append(row)

        # For each row of rooms
        for j in range(0, self.num_rows):
            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.room_grid[j][i]

                x_l, y_l = (room.top[0] + 1, room.top[1] + 1)
                x_m, y_m = (
                    room.top[0] + room.size[0] - 1,
                    room.top[1] + room.size[1] - 1,
                )

                # Door positions, order is right, down, left, up
                if i < self.num_cols - 1:
                    room.neighbors[0] = self.room_grid[j][i + 1]
                    room.door_pos[0] = (x_m, self._rand_int(y_l, y_m))
                if j < self.num_rows - 1:
                    room.neighbors[1] = self.room_grid[j + 1][i]
                    room.door_pos[1] = (self._rand_int(x_l, x_m), y_m)
                if i > 0:
                    room.neighbors[2] = self.room_grid[j][i - 1]
                    room.door_pos[2] = room.neighbors[2].door_pos[0]
                if j > 0:
                    room.neighbors[3] = self.room_grid[j - 1][i]
                    room.door_pos[3] = room.neighbors[3].door_pos[1]

        # The agent starts in the middle, facing right
        self.agent_pos = (
            (self.num_cols // 2) * (self.room_size - 1) + (self.room_size // 2),
            (self.num_rows // 2) * (self.room_size - 1) + (self.room_size // 2),
        )
        self.agent_dir = 0


class Level_GoToAfterPickUpLocal(LevelGen):
    """
    In order to test generalisation we only give to the agent the instruction:
    seq restricted to go to B after pickup A  with A and B without the following adj-noun pairs:
    - yellow box
    - red door/key
    - green ball
    - grey door
    (for memory issue our agent only used the past 3 observations)

    Competencies: Seq never seen in MixedTrainLocal
    """

    def __init__(
        self,
        room_size=8,
        num_rows=1,
        num_cols=1,
        num_dists=8,
        instr_kinds=["seq1"],
        locations=False,
        unblocking=False,
        implicit_unlock=False,
        **kwargs,
    ):
        action = "pick up seq pick up "

        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            action_kinds=[action],
            instr_kinds=instr_kinds,
            locations=locations,
            unblocking=unblocking,
            implicit_unlock=implicit_unlock,
            **kwargs,
        )

    def gen_mission(self):
        mission_accepted = False
        all_objects_reachable = False

        while not mission_accepted or not all_objects_reachable:
            self._regen_grid()
            self.place_agent()
            objs = self.add_distractors(
                num_distractors=self.num_dists + 2, all_unique=False
            )
            all_objects_reachable = self.check_objs_reachable(raise_exc=False)
            obj_a = self._rand_elem(objs)
            while str(obj_a.type) == "door":
                obj_a = self._rand_elem(objs)
            instr_a = PickupInstr(ObjDesc(obj_a.type, obj_a.color))
            obj_b = self._rand_elem(objs)
            if obj_a.type == obj_b.type and obj_a.color == obj_b.color:
                desc = ObjDesc(obj_a.type, obj_a.color)
                objas, poss = desc.find_matching_objs(self)
                if len(objas) < 2:
                    # if obj_a is the only object with this description obj_b has to be different
                    while obj_a.type == obj_b.type and obj_a.color == obj_b.color:
                        obj_b = self._rand_elem(objs)
            instr_b = GoToInstr(ObjDesc(obj_b.type, obj_b.color))

            self.instrs = AfterInstr(instr_b, instr_a)

            mission_accepted = not (self.exclude_substrings())

    def exclude_substrings(self):
        # True if contains excluded substring
        list_exclude_combinaison = [
            "yellow box",
            "red key",
            "red door",
            "green ball",
            "grey door",
        ]

        for sub_str in list_exclude_combinaison:
            if sub_str in self.instrs.surface(self):
                return True
        return False

    def _regen_grid(self):
        # Create the grid
        self.grid.grid = [None] * self.width * self.height

        # For each row of rooms
        for j in range(0, self.num_rows):
            row = []

            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.get_room(i, j)
                # suppress doors and objects
                room.doors = [None] * 4
                room.door_pos = [None] * 4
                room.neighbors = [None] * 4
                room.locked = False
                room.objs = []
                row.append(room)

                # Generate the walls for this room
                self.grid.wall_rect(*room.top, *room.size)

            self.room_grid.append(row)

        # For each row of rooms
        for j in range(0, self.num_rows):
            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.room_grid[j][i]

                x_l, y_l = (room.top[0] + 1, room.top[1] + 1)
                x_m, y_m = (
                    room.top[0] + room.size[0] - 1,
                    room.top[1] + room.size[1] - 1,
                )

                # Door positions, order is right, down, left, up
                if i < self.num_cols - 1:
                    room.neighbors[0] = self.room_grid[j][i + 1]
                    room.door_pos[0] = (x_m, self._rand_int(y_l, y_m))
                if j < self.num_rows - 1:
                    room.neighbors[1] = self.room_grid[j + 1][i]
                    room.door_pos[1] = (self._rand_int(x_l, x_m), y_m)
                if i > 0:
                    room.neighbors[2] = self.room_grid[j][i - 1]
                    room.door_pos[2] = room.neighbors[2].door_pos[0]
                if j > 0:
                    room.neighbors[3] = self.room_grid[j - 1][i]
                    room.door_pos[3] = room.neighbors[3].door_pos[1]

        # The agent starts in the middle, facing right
        self.agent_pos = (
            (self.num_cols // 2) * (self.room_size - 1) + (self.room_size // 2),
            (self.num_rows // 2) * (self.room_size - 1) + (self.room_size // 2),
        )
        self.agent_dir = 0
