"""
Copied and adapted from https://github.com/mila-iqia/babyai
"""

from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.roomgrid import Room
from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.core.verifier import (
    LOC_NAMES,
    OBJ_TYPES,
    OBJ_TYPES_NOT_DOOR,
    AfterInstr,
    AndInstr,
    BeforeInstr,
    GoToInstr,
    ObjDesc,
    OpenInstr,
    PickupInstr,
    PutNextInstr,
)


class LevelGen(RoomGridLevel):
    """
    Level generator which attempts to produce every possible sentence in
    the baby language as an instruction.
    """

    def __init__(
        self,
        room_size=8,
        num_rows=3,
        num_cols=3,
        num_dists=18,
        locked_room_prob=0.5,
        locations=True,
        unblocking=True,
        implicit_unlock=True,
        action_kinds=("goto", "pickup", "open", "putnext"),
        instr_kinds=("action", "and", "seq"),
        **kwargs,
    ):
        self.num_dists = num_dists
        self.locked_room_prob = locked_room_prob
        self.locations = locations
        self.unblocking = unblocking
        self.implicit_unlock = implicit_unlock
        self.action_kinds = action_kinds
        self.instr_kinds = instr_kinds

        self.locked_room = None

        super().__init__(
            room_size=room_size, num_rows=num_rows, num_cols=num_cols, **kwargs
        )

    def gen_mission(self):
        if self._rand_float(0, 1) < self.locked_room_prob:
            self.add_locked_room()

        self.connect_all()

        self.add_distractors(num_distractors=self.num_dists, all_unique=False)

        # The agent must be placed after all the object to respect constraints
        while True:
            self.place_agent()
            start_room = self.room_from_pos(*self.agent_pos)
            # Ensure that we are not placing the agent in the locked room
            if start_room is self.locked_room:
                continue
            break

        # If no unblocking required, make sure all objects are
        # reachable without unblocking
        if not self.unblocking:
            self.check_objs_reachable()

        # Generate random instructions
        self.instrs = self.rand_instr(
            action_kinds=self.action_kinds, instr_kinds=self.instr_kinds
        )

    def add_locked_room(self, color=None):
        # Until we've successfully added a locked room
        while True:
            i = self._rand_int(0, self.num_cols)
            j = self._rand_int(0, self.num_rows)
            door_idx = self._rand_int(0, 4)
            self.locked_room = self.get_room(i, j)

            # Don't add a locked door in an external wall
            if self.locked_room.neighbors[door_idx] is None:
                continue

            if color is not None:
                door, _ = self.add_door(i, j, door_idx, color=color, locked=True)
            else:
                door, _ = self.add_door(i, j, door_idx, locked=True)

            # Done adding locked room
            break

        # Until we find a room to put the key
        while True:
            i = self._rand_int(0, self.num_cols)
            j = self._rand_int(0, self.num_rows)
            key_room = self.get_room(i, j)

            if key_room is self.locked_room:
                continue

            self.add_object(i, j, "key", door.color)
            break

    def rand_obj(self, types=OBJ_TYPES, colors=COLOR_NAMES, max_tries=100):
        """
        Generate a random object descriptor
        """

        num_tries = 0

        # Keep trying until we find a matching object
        while True:
            if num_tries > max_tries:
                raise RecursionError("failed to find suitable object")
            num_tries += 1

            color = self._rand_elem([None, *colors])
            type = self._rand_elem(types)

            loc = None
            if self.locations and self._rand_bool():
                loc = self._rand_elem(LOC_NAMES)

            desc = ObjDesc(type, color, loc)

            # Find all objects matching the descriptor
            objs, poss = desc.find_matching_objs(self)

            # The description must match at least one object
            if len(objs) == 0:
                continue

            # If no implicit unlocking is required
            if not self.implicit_unlock and isinstance(self.locked_room, Room):
                locked_room = self.locked_room
                # Check that at least one object is not in the locked room
                pos_not_locked = list(
                    filter(lambda p: not locked_room.pos_inside(*p), poss)
                )

                if len(pos_not_locked) == 0:
                    continue

            # Found a valid object description
            return desc

    def rand_instr(self, action_kinds, instr_kinds, depth=0):
        """
        Generate random instructions
        """

        kind = self._rand_elem(instr_kinds)

        if kind == "action":
            action = self._rand_elem(action_kinds)

            if action == "goto":
                return GoToInstr(self.rand_obj())
            elif action == "pickup":
                return PickupInstr(self.rand_obj(types=OBJ_TYPES_NOT_DOOR))
            elif action == "open":
                return OpenInstr(self.rand_obj(types=["door"]))
            elif action == "putnext":
                return PutNextInstr(
                    self.rand_obj(types=OBJ_TYPES_NOT_DOOR), self.rand_obj()
                )

            assert False

        elif kind == "and":
            instr_a = self.rand_instr(
                action_kinds=action_kinds, instr_kinds=["action"], depth=depth + 1
            )
            instr_b = self.rand_instr(
                action_kinds=action_kinds, instr_kinds=["action"], depth=depth + 1
            )
            return AndInstr(instr_a, instr_b)

        elif kind == "seq":
            instr_a = self.rand_instr(
                action_kinds=action_kinds,
                instr_kinds=["action", "and"],
                depth=depth + 1,
            )
            instr_b = self.rand_instr(
                action_kinds=action_kinds,
                instr_kinds=["action", "and"],
                depth=depth + 1,
            )

            kind = self._rand_elem(["before", "after"])

            if kind == "before":
                return BeforeInstr(instr_a, instr_b)
            elif kind == "after":
                return AfterInstr(instr_a, instr_b)

            assert False

        assert False
