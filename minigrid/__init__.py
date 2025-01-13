from __future__ import annotations

from gymnasium.envs.registration import register

from minigrid import minigrid_env, wrappers
from minigrid.core import roomgrid
from minigrid.core.world_object import Wall
from minigrid.envs.wfc.config import WFC_PRESETS, register_wfc_presets

__version__ = "2.5.0"


def register_minigrid_envs():
    # BlockedUnlockPickup
    # ----------------------------------------

    register(
        id="MiniGrid-BlockedUnlockPickup-v0",
        entry_point="minigrid.envs:BlockedUnlockPickupEnv",
    )

    # LavaCrossing
    # ----------------------------------------
    register(
        id="MiniGrid-LavaCrossingS9N1-v0",
        entry_point="minigrid.envs:CrossingEnv",
        kwargs={"size": 9, "num_crossings": 1},
    )

    register(
        id="MiniGrid-LavaCrossingS9N2-v0",
        entry_point="minigrid.envs:CrossingEnv",
        kwargs={"size": 9, "num_crossings": 2},
    )

    register(
        id="MiniGrid-LavaCrossingS9N3-v0",
        entry_point="minigrid.envs:CrossingEnv",
        kwargs={"size": 9, "num_crossings": 3},
    )

    register(
        id="MiniGrid-LavaCrossingS11N5-v0",
        entry_point="minigrid.envs:CrossingEnv",
        kwargs={"size": 11, "num_crossings": 5},
    )

    # SimpleCrossing
    # ----------------------------------------

    register(
        id="MiniGrid-SimpleCrossingS9N1-v0",
        entry_point="minigrid.envs:CrossingEnv",
        kwargs={"size": 9, "num_crossings": 1, "obstacle_type": Wall},
    )

    register(
        id="MiniGrid-SimpleCrossingS9N2-v0",
        entry_point="minigrid.envs:CrossingEnv",
        kwargs={"size": 9, "num_crossings": 2, "obstacle_type": Wall},
    )

    register(
        id="MiniGrid-SimpleCrossingS9N3-v0",
        entry_point="minigrid.envs:CrossingEnv",
        kwargs={"size": 9, "num_crossings": 3, "obstacle_type": Wall},
    )

    register(
        id="MiniGrid-SimpleCrossingS11N5-v0",
        entry_point="minigrid.envs:CrossingEnv",
        kwargs={"size": 11, "num_crossings": 5, "obstacle_type": Wall},
    )

    # DistShift
    # ----------------------------------------

    register(
        id="MiniGrid-DistShift1-v0",
        entry_point="minigrid.envs:DistShiftEnv",
        kwargs={"strip2_row": 2},
    )

    register(
        id="MiniGrid-DistShift2-v0",
        entry_point="minigrid.envs:DistShiftEnv",
        kwargs={"strip2_row": 5},
    )

    # DoorKey
    # ----------------------------------------

    register(
        id="MiniGrid-DoorKey-5x5-v0",
        entry_point="minigrid.envs:DoorKeyEnv",
        kwargs={"size": 5},
    )

    register(
        id="MiniGrid-DoorKey-6x6-v0",
        entry_point="minigrid.envs:DoorKeyEnv",
        kwargs={"size": 6},
    )

    register(
        id="MiniGrid-DoorKey-8x8-v0",
        entry_point="minigrid.envs:DoorKeyEnv",
        kwargs={"size": 8},
    )

    register(
        id="MiniGrid-DoorKey-16x16-v0",
        entry_point="minigrid.envs:DoorKeyEnv",
        kwargs={"size": 16},
    )

    # Dynamic-Obstacles
    # ----------------------------------------

    register(
        id="MiniGrid-Dynamic-Obstacles-5x5-v0",
        entry_point="minigrid.envs:DynamicObstaclesEnv",
        kwargs={"size": 5, "n_obstacles": 2},
    )

    register(
        id="MiniGrid-Dynamic-Obstacles-Random-5x5-v0",
        entry_point="minigrid.envs:DynamicObstaclesEnv",
        kwargs={"size": 5, "agent_start_pos": None, "n_obstacles": 2},
    )

    register(
        id="MiniGrid-Dynamic-Obstacles-6x6-v0",
        entry_point="minigrid.envs:DynamicObstaclesEnv",
        kwargs={"size": 6, "n_obstacles": 3},
    )

    register(
        id="MiniGrid-Dynamic-Obstacles-Random-6x6-v0",
        entry_point="minigrid.envs:DynamicObstaclesEnv",
        kwargs={"size": 6, "agent_start_pos": None, "n_obstacles": 3},
    )

    register(
        id="MiniGrid-Dynamic-Obstacles-8x8-v0",
        entry_point="minigrid.envs:DynamicObstaclesEnv",
    )

    register(
        id="MiniGrid-Dynamic-Obstacles-16x16-v0",
        entry_point="minigrid.envs:DynamicObstaclesEnv",
        kwargs={"size": 16, "n_obstacles": 8},
    )

    # Empty
    # ----------------------------------------

    register(
        id="MiniGrid-Empty-5x5-v0",
        entry_point="minigrid.envs:EmptyEnv",
        kwargs={"size": 5},
    )

    register(
        id="MiniGrid-Empty-Random-5x5-v0",
        entry_point="minigrid.envs:EmptyEnv",
        kwargs={"size": 5, "agent_start_pos": None},
    )

    register(
        id="MiniGrid-Empty-6x6-v0",
        entry_point="minigrid.envs:EmptyEnv",
        kwargs={"size": 6},
    )

    register(
        id="MiniGrid-Empty-Random-6x6-v0",
        entry_point="minigrid.envs:EmptyEnv",
        kwargs={"size": 6, "agent_start_pos": None},
    )

    register(
        id="MiniGrid-Empty-8x8-v0",
        entry_point="minigrid.envs:EmptyEnv",
    )

    register(
        id="MiniGrid-Empty-16x16-v0",
        entry_point="minigrid.envs:EmptyEnv",
        kwargs={"size": 16},
    )

    # Fetch
    # ----------------------------------------

    register(
        id="MiniGrid-Fetch-5x5-N2-v0",
        entry_point="minigrid.envs:FetchEnv",
        kwargs={"size": 5, "numObjs": 2},
    )

    register(
        id="MiniGrid-Fetch-6x6-N2-v0",
        entry_point="minigrid.envs:FetchEnv",
        kwargs={"size": 6, "numObjs": 2},
    )

    register(id="MiniGrid-Fetch-8x8-N3-v0", entry_point="minigrid.envs:FetchEnv")

    # FourRooms
    # ----------------------------------------

    register(
        id="MiniGrid-FourRooms-v0",
        entry_point="minigrid.envs:FourRoomsEnv",
    )

    # GoToDoor
    # ----------------------------------------

    register(
        id="MiniGrid-GoToDoor-5x5-v0",
        entry_point="minigrid.envs:GoToDoorEnv",
    )

    register(
        id="MiniGrid-GoToDoor-6x6-v0",
        entry_point="minigrid.envs:GoToDoorEnv",
        kwargs={"size": 6},
    )

    register(
        id="MiniGrid-GoToDoor-8x8-v0",
        entry_point="minigrid.envs:GoToDoorEnv",
        kwargs={"size": 8},
    )

    # GoToObject
    # ----------------------------------------

    register(
        id="MiniGrid-GoToObject-6x6-N2-v0",
        entry_point="minigrid.envs:GoToObjectEnv",
    )

    register(
        id="MiniGrid-GoToObject-8x8-N2-v0",
        entry_point="minigrid.envs:GoToObjectEnv",
        kwargs={"size": 8, "numObjs": 2},
    )

    # KeyCorridor
    # ----------------------------------------

    register(
        id="MiniGrid-KeyCorridorS3R1-v0",
        entry_point="minigrid.envs:KeyCorridorEnv",
        kwargs={"room_size": 3, "num_rows": 1},
    )

    register(
        id="MiniGrid-KeyCorridorS3R2-v0",
        entry_point="minigrid.envs:KeyCorridorEnv",
        kwargs={"room_size": 3, "num_rows": 2},
    )

    register(
        id="MiniGrid-KeyCorridorS3R3-v0",
        entry_point="minigrid.envs:KeyCorridorEnv",
        kwargs={"room_size": 3, "num_rows": 3},
    )

    register(
        id="MiniGrid-KeyCorridorS4R3-v0",
        entry_point="minigrid.envs:KeyCorridorEnv",
        kwargs={"room_size": 4, "num_rows": 3},
    )

    register(
        id="MiniGrid-KeyCorridorS5R3-v0",
        entry_point="minigrid.envs:KeyCorridorEnv",
        kwargs={"room_size": 5, "num_rows": 3},
    )

    register(
        id="MiniGrid-KeyCorridorS6R3-v0",
        entry_point="minigrid.envs:KeyCorridorEnv",
        kwargs={"room_size": 6, "num_rows": 3},
    )

    # LavaGap
    # ----------------------------------------

    register(
        id="MiniGrid-LavaGapS5-v0",
        entry_point="minigrid.envs:LavaGapEnv",
        kwargs={"size": 5},
    )

    register(
        id="MiniGrid-LavaGapS6-v0",
        entry_point="minigrid.envs:LavaGapEnv",
        kwargs={"size": 6},
    )

    register(
        id="MiniGrid-LavaGapS7-v0",
        entry_point="minigrid.envs:LavaGapEnv",
        kwargs={"size": 7},
    )

    # LockedRoom
    # ----------------------------------------

    register(
        id="MiniGrid-LockedRoom-v0",
        entry_point="minigrid.envs:LockedRoomEnv",
    )

    # Memory
    # ----------------------------------------

    register(
        id="MiniGrid-MemoryS17Random-v0",
        entry_point="minigrid.envs:MemoryEnv",
        kwargs={"size": 17, "random_length": True},
    )

    register(
        id="MiniGrid-MemoryS13Random-v0",
        entry_point="minigrid.envs:MemoryEnv",
        kwargs={"size": 13, "random_length": True},
    )

    register(
        id="MiniGrid-MemoryS13-v0",
        entry_point="minigrid.envs:MemoryEnv",
        kwargs={"size": 13},
    )

    register(
        id="MiniGrid-MemoryS11-v0",
        entry_point="minigrid.envs:MemoryEnv",
        kwargs={"size": 11},
    )

    register(
        id="MiniGrid-MemoryS9-v0",
        entry_point="minigrid.envs:MemoryEnv",
        kwargs={"size": 9},
    )

    register(
        id="MiniGrid-MemoryS7-v0",
        entry_point="minigrid.envs:MemoryEnv",
        kwargs={"size": 7},
    )

    # MultiRoom
    # ----------------------------------------

    register(
        id="MiniGrid-MultiRoom-N2-S4-v0",
        entry_point="minigrid.envs:MultiRoomEnv",
        kwargs={"minNumRooms": 2, "maxNumRooms": 2, "maxRoomSize": 4},
    )

    register(
        id="MiniGrid-MultiRoom-N4-S5-v0",
        entry_point="minigrid.envs:MultiRoomEnv",
        kwargs={"minNumRooms": 6, "maxNumRooms": 6, "maxRoomSize": 5},
    )

    register(
        id="MiniGrid-MultiRoom-N6-v0",
        entry_point="minigrid.envs:MultiRoomEnv",
        kwargs={"minNumRooms": 6, "maxNumRooms": 6},
    )

    # ObstructedMaze
    # ----------------------------------------

    register(
        id="MiniGrid-ObstructedMaze-1Dl-v0",
        entry_point="minigrid.envs:ObstructedMaze_1Dlhb",
        kwargs={"key_in_box": False, "blocked": False},
    )

    register(
        id="MiniGrid-ObstructedMaze-1Dlh-v0",
        entry_point="minigrid.envs:ObstructedMaze_1Dlhb",
        kwargs={"key_in_box": True, "blocked": False},
    )

    register(
        id="MiniGrid-ObstructedMaze-1Dlhb-v0",
        entry_point="minigrid.envs:ObstructedMaze_1Dlhb",
    )

    register(
        id="MiniGrid-ObstructedMaze-2Dl-v0",
        entry_point="minigrid.envs:ObstructedMaze_Full",
        kwargs={
            "agent_room": (2, 1),
            "key_in_box": False,
            "blocked": False,
            "num_quarters": 1,
            "num_rooms_visited": 4,
        },
    )

    register(
        id="MiniGrid-ObstructedMaze-2Dlh-v0",
        entry_point="minigrid.envs:ObstructedMaze_Full",
        kwargs={
            "agent_room": (2, 1),
            "key_in_box": True,
            "blocked": False,
            "num_quarters": 1,
            "num_rooms_visited": 4,
        },
    )

    register(
        id="MiniGrid-ObstructedMaze-2Dlhb-v0",
        entry_point="minigrid.envs:ObstructedMaze_Full",
        kwargs={
            "agent_room": (2, 1),
            "key_in_box": True,
            "blocked": True,
            "num_quarters": 1,
            "num_rooms_visited": 4,
        },
    )

    register(
        id="MiniGrid-ObstructedMaze-1Q-v0",
        entry_point="minigrid.envs:ObstructedMaze_Full",
        kwargs={
            "agent_room": (1, 1),
            "key_in_box": True,
            "blocked": True,
            "num_quarters": 1,
            "num_rooms_visited": 5,
        },
    )

    register(
        id="MiniGrid-ObstructedMaze-2Q-v0",
        entry_point="minigrid.envs:ObstructedMaze_Full",
        kwargs={
            "agent_room": (2, 1),
            "key_in_box": True,
            "blocked": True,
            "num_quarters": 2,
            "num_rooms_visited": 11,
        },
    )

    register(
        id="MiniGrid-ObstructedMaze-Full-v0",
        entry_point="minigrid.envs:ObstructedMaze_Full",
    )

    # ObstructedMaze-v1
    # ----------------------------------------

    register(
        id="MiniGrid-ObstructedMaze-2Dlhb-v1",
        entry_point="minigrid.envs:ObstructedMaze_Full_V1",
        kwargs={
            "agent_room": (2, 1),
            "key_in_box": True,
            "blocked": True,
            "num_quarters": 1,
            "num_rooms_visited": 4,
        },
    )

    register(
        id="MiniGrid-ObstructedMaze-1Q-v1",
        entry_point="minigrid.envs:ObstructedMaze_Full_V1",
        kwargs={
            "agent_room": (1, 1),
            "key_in_box": True,
            "blocked": True,
            "num_quarters": 1,
            "num_rooms_visited": 5,
        },
    )

    register(
        id="MiniGrid-ObstructedMaze-2Q-v1",
        entry_point="minigrid.envs:ObstructedMaze_Full_V1",
        kwargs={
            "agent_room": (2, 1),
            "key_in_box": True,
            "blocked": True,
            "num_quarters": 2,
            "num_rooms_visited": 11,
        },
    )

    register(
        id="MiniGrid-ObstructedMaze-Full-v1",
        entry_point="minigrid.envs:ObstructedMaze_Full_V1",
    )

    # Playground
    # ----------------------------------------

    register(
        id="MiniGrid-Playground-v0",
        entry_point="minigrid.envs:PlaygroundEnv",
    )

    # PutNear
    # ----------------------------------------

    register(
        id="MiniGrid-PutNear-6x6-N2-v0",
        entry_point="minigrid.envs:PutNearEnv",
    )

    register(
        id="MiniGrid-PutNear-8x8-N3-v0",
        entry_point="minigrid.envs:PutNearEnv",
        kwargs={"size": 8, "numObjs": 3},
    )

    # RedBlueDoors
    # ----------------------------------------

    register(
        id="MiniGrid-RedBlueDoors-6x6-v0",
        entry_point="minigrid.envs:RedBlueDoorEnv",
        kwargs={"size": 6},
    )

    register(
        id="MiniGrid-RedBlueDoors-8x8-v0",
        entry_point="minigrid.envs:RedBlueDoorEnv",
    )

    # Unlock
    # ----------------------------------------

    register(id="MiniGrid-Unlock-v0", entry_point="minigrid.envs:UnlockEnv")

    # UnlockPickup
    # ----------------------------------------

    register(
        id="MiniGrid-UnlockPickup-v0",
        entry_point="minigrid.envs:UnlockPickupEnv",
    )

    # WaveFunctionCollapse
    # ----------------------------------------
    register_wfc_presets(WFC_PRESETS, register)

    # BabyAI - Language based levels - GoTo
    # ----------------------------------------

    register(
        id="BabyAI-GoToRedBallGrey-v0",
        entry_point="minigrid.envs.babyai:GoToRedBallGrey",
    )

    register(
        id="BabyAI-GoToRedBall-v0",
        entry_point="minigrid.envs.babyai:GoToRedBall",
    )

    register(
        id="BabyAI-GoToRedBallNoDists-v0",
        entry_point="minigrid.envs.babyai:GoToRedBallNoDists",
    )

    register(
        id="BabyAI-GoToObj-v0",
        entry_point="minigrid.envs.babyai:GoToObj",
    )

    register(
        id="BabyAI-GoToObjS4-v0",
        entry_point="minigrid.envs.babyai:GoToObj",
        kwargs={"room_size": 4},
    )

    register(
        id="BabyAI-GoToObjS6-v1",
        entry_point="minigrid.envs.babyai:GoToObj",
        kwargs={"room_size": 6},
    )

    register(
        id="BabyAI-GoToLocal-v0",
        entry_point="minigrid.envs.babyai:GoToLocal",
    )

    register(
        id="BabyAI-GoToLocalS5N2-v0",
        entry_point="minigrid.envs.babyai:GoToLocal",
        kwargs={"room_size": 5, "num_dists": 2},
    )

    register(
        id="BabyAI-GoToLocalS6N2-v0",
        entry_point="minigrid.envs.babyai:GoToLocal",
        kwargs={"room_size": 6, "num_dists": 2},
    )

    register(
        id="BabyAI-GoToLocalS6N3-v0",
        entry_point="minigrid.envs.babyai:GoToLocal",
        kwargs={"room_size": 6, "num_dists": 3},
    )

    register(
        id="BabyAI-GoToLocalS6N4-v0",
        entry_point="minigrid.envs.babyai:GoToLocal",
        kwargs={"room_size": 6, "num_dists": 4},
    )

    register(
        id="BabyAI-GoToLocalS7N4-v0",
        entry_point="minigrid.envs.babyai:GoToLocal",
        kwargs={"room_size": 7, "num_dists": 4},
    )

    register(
        id="BabyAI-GoToLocalS7N5-v0",
        entry_point="minigrid.envs.babyai:GoToLocal",
        kwargs={"room_size": 7, "num_dists": 5},
    )

    register(
        id="BabyAI-GoToLocalS8N2-v0",
        entry_point="minigrid.envs.babyai:GoToLocal",
        kwargs={"room_size": 8, "num_dists": 2},
    )

    register(
        id="BabyAI-GoToLocalS8N3-v0",
        entry_point="minigrid.envs.babyai:GoToLocal",
        kwargs={"room_size": 8, "num_dists": 3},
    )

    register(
        id="BabyAI-GoToLocalS8N4-v0",
        entry_point="minigrid.envs.babyai:GoToLocal",
        kwargs={"room_size": 8, "num_dists": 4},
    )

    register(
        id="BabyAI-GoToLocalS8N5-v0",
        entry_point="minigrid.envs.babyai:GoToLocal",
        kwargs={"room_size": 8, "num_dists": 5},
    )

    register(
        id="BabyAI-GoToLocalS8N6-v0",
        entry_point="minigrid.envs.babyai:GoToLocal",
        kwargs={"room_size": 8, "num_dists": 6},
    )

    register(
        id="BabyAI-GoToLocalS8N7-v0",
        entry_point="minigrid.envs.babyai:GoToLocal",
        kwargs={"room_size": 8, "num_dists": 7},
    )

    register(
        id="BabyAI-GoTo-v0",
        entry_point="minigrid.envs.babyai:GoTo",
    )

    register(
        "BabyAI-GoToOpen-v0",
        entry_point="minigrid.envs.babyai:GoTo",
        kwargs={"doors_open": True},
    )

    register(
        id="BabyAI-GoToObjMaze-v0",
        entry_point="minigrid.envs.babyai:GoTo",
        kwargs={"num_dists": 1, "doors_open": False},
    )

    register(
        id="BabyAI-GoToObjMazeOpen-v0",
        entry_point="minigrid.envs.babyai:GoTo",
        kwargs={"num_dists": 1, "doors_open": True},
    )

    register(
        id="BabyAI-GoToObjMazeS4R2-v0",
        entry_point="minigrid.envs.babyai:GoTo",
        kwargs={"num_dists": 1, "room_size": 4, "num_rows": 2, "num_cols": 2},
    )

    register(
        id="BabyAI-GoToObjMazeS4-v0",
        entry_point="minigrid.envs.babyai:GoTo",
        kwargs={"num_dists": 1, "room_size": 4},
    )

    register(
        id="BabyAI-GoToObjMazeS5-v0",
        entry_point="minigrid.envs.babyai:GoTo",
        kwargs={"num_dists": 1, "room_size": 5},
    )

    register(
        id="BabyAI-GoToObjMazeS6-v0",
        entry_point="minigrid.envs.babyai:GoTo",
        kwargs={"num_dists": 1, "room_size": 6},
    )

    register(
        id="BabyAI-GoToObjMazeS7-v0",
        entry_point="minigrid.envs.babyai:GoTo",
        kwargs={"num_dists": 1, "room_size": 7},
    )

    register(
        id="BabyAI-GoToImpUnlock-v0",
        entry_point="minigrid.envs.babyai:GoToImpUnlock",
    )

    register(
        id="BabyAI-GoToSeq-v0",
        entry_point="minigrid.envs.babyai:GoToSeq",
    )

    register(
        id="BabyAI-GoToSeqS5R2-v0",
        entry_point="minigrid.envs.babyai:GoToSeq",
        kwargs={"room_size": 5, "num_rows": 2, "num_cols": 2, "num_dists": 4},
    )

    register(
        id="BabyAI-GoToRedBlueBall-v0",
        entry_point="minigrid.envs.babyai:GoToRedBlueBall",
    )

    register(
        id="BabyAI-GoToDoor-v0",
        entry_point="minigrid.envs.babyai:GoToDoor",
    )

    register(
        id="BabyAI-GoToObjDoor-v0",
        entry_point="minigrid.envs.babyai:GoToObjDoor",
    )

    # BabyAI - Language based levels - Open
    # ----------------------------------------

    register(
        id="BabyAI-Open-v0",
        entry_point="minigrid.envs.babyai:Open",
    )

    register(
        id="BabyAI-OpenRedDoor-v0",
        entry_point="minigrid.envs.babyai:OpenRedDoor",
    )

    register(
        id="BabyAI-OpenDoor-v0",
        entry_point="minigrid.envs.babyai:OpenDoor",
    )

    register(
        id="BabyAI-OpenDoorDebug-v0",
        entry_point="minigrid.envs.babyai:OpenDoor",
        kwargs={"debug": True, "select_by": None},
    )

    register(
        id="BabyAI-OpenDoorColor-v0",
        entry_point="minigrid.envs.babyai:OpenDoor",
        kwargs={"select_by": "color"},
    )

    register(
        id="BabyAI-OpenDoorLoc-v0",
        entry_point="minigrid.envs.babyai:OpenDoor",
        kwargs={"select_by": "loc"},
    )

    register(
        id="BabyAI-OpenTwoDoors-v0",
        entry_point="minigrid.envs.babyai:OpenTwoDoors",
    )

    register(
        id="BabyAI-OpenRedBlueDoors-v0",
        entry_point="minigrid.envs.babyai:OpenTwoDoors",
        kwargs={"first_color": "red", "second_color": "blue"},
    )

    register(
        id="BabyAI-OpenRedBlueDoorsDebug-v0",
        entry_point="minigrid.envs.babyai:OpenTwoDoors",
        kwargs={
            "first_color": "red",
            "second_color": "blue",
            "strict": True,
        },
    )

    register(
        id="BabyAI-OpenDoorsOrderN2-v0",
        entry_point="minigrid.envs.babyai:OpenDoorsOrder",
        kwargs={"num_doors": 2},
    )

    register(
        id="BabyAI-OpenDoorsOrderN4-v0",
        entry_point="minigrid.envs.babyai:OpenDoorsOrder",
        kwargs={"num_doors": 4},
    )

    register(
        id="BabyAI-OpenDoorsOrderN2Debug-v0",
        entry_point="minigrid.envs.babyai:OpenDoorsOrder",
        kwargs={"debug": True, "num_doors": 2},
    )

    register(
        id="BabyAI-OpenDoorsOrderN4Debug-v0",
        entry_point="minigrid.envs.babyai:OpenDoorsOrder",
        kwargs={"debug": True, "num_doors": 4},
    )

    # BabyAI - Language based levels - Pickup
    # ----------------------------------------

    register(
        id="BabyAI-Pickup-v0",
        entry_point="minigrid.envs.babyai:Pickup",
    )

    register(
        id="BabyAI-UnblockPickup-v0",
        entry_point="minigrid.envs.babyai:UnblockPickup",
    )

    register(
        id="BabyAI-PickupLoc-v0",
        entry_point="minigrid.envs.babyai:PickupLoc",
    )

    register(
        id="BabyAI-PickupDist-v0",
        entry_point="minigrid.envs.babyai:PickupDist",
    )

    register(
        id="BabyAI-PickupDistDebug-v0",
        entry_point="minigrid.envs.babyai:PickupDist",
        kwargs={"debug": True},
    )

    register(
        id="BabyAI-PickupAbove-v0",
        entry_point="minigrid.envs.babyai:PickupAbove",
    )

    # BabyAI - Language based levels - PutNext
    # ----------------------------------------

    register(
        id="BabyAI-PutNextLocal-v0",
        entry_point="minigrid.envs.babyai:PutNextLocal",
    )

    register(
        id="BabyAI-PutNextLocalS5N3-v0",
        entry_point="minigrid.envs.babyai:PutNextLocal",
        kwargs={"room_size": 5, "num_objs": 3},
    )

    register(
        id="BabyAI-PutNextLocalS6N4-v0",
        entry_point="minigrid.envs.babyai:PutNextLocal",
        kwargs={"room_size": 6, "num_objs": 4},
    )

    register(
        id="BabyAI-PutNextS4N1-v0",
        entry_point="minigrid.envs.babyai:PutNext",
        kwargs={"room_size": 4, "objs_per_room": 1},
    )

    register(
        id="BabyAI-PutNextS5N2-v0",
        entry_point="minigrid.envs.babyai:PutNext",
        kwargs={"room_size": 5, "objs_per_room": 2},
    )

    register(
        id="BabyAI-PutNextS5N1-v0",
        entry_point="minigrid.envs.babyai:PutNext",
        kwargs={"room_size": 5, "objs_per_room": 1},
    )

    register(
        id="BabyAI-PutNextS6N3-v0",
        entry_point="minigrid.envs.babyai:PutNext",
        kwargs={"room_size": 6, "objs_per_room": 3},
    )

    register(
        id="BabyAI-PutNextS7N4-v0",
        entry_point="minigrid.envs.babyai:PutNext",
        kwargs={"room_size": 7, "objs_per_room": 4},
    )

    register(
        id="BabyAI-PutNextS5N2Carrying-v0",
        entry_point="minigrid.envs.babyai:PutNext",
        kwargs={"room_size": 5, "objs_per_room": 2, "start_carrying": True},
    )

    register(
        id="BabyAI-PutNextS6N3Carrying-v0",
        entry_point="minigrid.envs.babyai:PutNext",
        kwargs={"room_size": 6, "objs_per_room": 3, "start_carrying": True},
    )

    register(
        id="BabyAI-PutNextS7N4Carrying-v0",
        entry_point="minigrid.envs.babyai:PutNext",
        kwargs={"room_size": 7, "objs_per_room": 4, "start_carrying": True},
    )

    # BabyAI - Language based levels - Unlock
    # ----------------------------------------

    register(
        id="BabyAI-Unlock-v0",
        entry_point="minigrid.envs.babyai:Unlock",
    )

    register(
        id="BabyAI-UnlockLocal-v0",
        entry_point="minigrid.envs.babyai:UnlockLocal",
    )

    register(
        id="BabyAI-UnlockLocalDist-v0",
        entry_point="minigrid.envs.babyai:UnlockLocal",
        kwargs={"distractors": True},
    )

    register(
        id="BabyAI-KeyInBox-v0",
        entry_point="minigrid.envs.babyai:KeyInBox",
    )

    register(
        id="BabyAI-UnlockPickup-v0",
        entry_point="minigrid.envs.babyai:UnlockPickup",
    )

    register(
        id="BabyAI-UnlockPickupDist-v0",
        entry_point="minigrid.envs.babyai:UnlockPickup",
        kwargs={"distractors": True},
    )

    register(
        id="BabyAI-BlockedUnlockPickup-v0",
        entry_point="minigrid.envs.babyai:BlockedUnlockPickup",
    )

    register(
        id="BabyAI-UnlockToUnlock-v0",
        entry_point="minigrid.envs.babyai:UnlockToUnlock",
    )

    # BabyAI - Language based levels - Other
    # ----------------------------------------

    register(
        id="BabyAI-ActionObjDoor-v0",
        entry_point="minigrid.envs.babyai:ActionObjDoor",
    )

    register(
        id="BabyAI-FindObjS5-v0",
        entry_point="minigrid.envs.babyai:FindObjS5",
    )

    register(
        id="BabyAI-FindObjS6-v0",
        entry_point="minigrid.envs.babyai:FindObjS5",
        kwargs={"room_size": 6},
    )

    register(
        id="BabyAI-FindObjS7-v0",
        entry_point="minigrid.envs.babyai:FindObjS5",
        kwargs={"room_size": 7},
    )

    register(
        id="BabyAI-KeyCorridor-v0",
        entry_point="minigrid.envs.babyai:KeyCorridor",
    )

    register(
        id="BabyAI-KeyCorridorS3R1-v0",
        entry_point="minigrid.envs.babyai:KeyCorridor",
        kwargs={"room_size": 3, "num_rows": 1},
    )

    register(
        id="BabyAI-KeyCorridorS3R2-v0",
        entry_point="minigrid.envs.babyai:KeyCorridor",
        kwargs={"room_size": 3, "num_rows": 2},
    )

    register(
        id="BabyAI-KeyCorridorS3R3-v0",
        entry_point="minigrid.envs.babyai:KeyCorridor",
        kwargs={"room_size": 3, "num_rows": 3},
    )

    register(
        id="BabyAI-KeyCorridorS4R3-v0",
        entry_point="minigrid.envs.babyai:KeyCorridor",
        kwargs={"room_size": 4, "num_rows": 3},
    )

    register(
        id="BabyAI-KeyCorridorS5R3-v0",
        entry_point="minigrid.envs.babyai:KeyCorridor",
        kwargs={"room_size": 5, "num_rows": 3},
    )

    register(
        id="BabyAI-KeyCorridorS6R3-v0",
        entry_point="minigrid.envs.babyai:KeyCorridor",
        kwargs={"room_size": 6, "num_rows": 3},
    )

    register(
        id="BabyAI-OneRoomS8-v0",
        entry_point="minigrid.envs.babyai:OneRoomS8",
    )

    register(
        id="BabyAI-OneRoomS12-v0",
        entry_point="minigrid.envs.babyai:OneRoomS8",
        kwargs={"room_size": 12},
    )

    register(
        id="BabyAI-OneRoomS16-v0",
        entry_point="minigrid.envs.babyai:OneRoomS8",
        kwargs={"room_size": 16},
    )

    register(
        id="BabyAI-OneRoomS20-v0",
        entry_point="minigrid.envs.babyai:OneRoomS8",
        kwargs={"room_size": 20},
    )

    register(
        id="BabyAI-MoveTwoAcrossS5N2-v0",
        entry_point="minigrid.envs.babyai:MoveTwoAcross",
        kwargs={"room_size": 5, "objs_per_room": 2},
    )

    register(
        id="BabyAI-MoveTwoAcrossS8N9-v0",
        entry_point="minigrid.envs.babyai:MoveTwoAcross",
        kwargs={"room_size": 8, "objs_per_room": 9},
    )

    # BabyAI - Language based levels - Synth
    # ----------------------------------------

    register(
        id="BabyAI-Synth-v0",
        entry_point="minigrid.envs.babyai:Synth",
    )

    register(
        id="BabyAI-SynthS5R2-v0",
        entry_point="minigrid.envs.babyai:Synth",
        kwargs={"room_size": 5, "num_rows": 2},
    )

    register(
        id="BabyAI-SynthLoc-v0",
        entry_point="minigrid.envs.babyai:SynthLoc",
    )

    register(
        id="BabyAI-SynthSeq-v0",
        entry_point="minigrid.envs.babyai:SynthSeq",
    )

    register(
        id="BabyAI-MiniBossLevel-v0",
        entry_point="minigrid.envs.babyai:MiniBossLevel",
    )

    register(
        id="BabyAI-BossLevel-v0",
        entry_point="minigrid.envs.babyai:BossLevel",
    )

    register(
        id="BabyAI-BossLevelNoUnlock-v0",
        entry_point="minigrid.envs.babyai:BossLevelNoUnlock",
    )

    # BabyAI - Language based levels - Level_MixedTrainLocal and Level_MixedTestLocal
    # ----------------------------------------

    register(
        id="BabyAI-MixedTrainLocal-v0",
        entry_point="minigrid.envs.babyai:Level_MixedTrainLocal",
    )

    register(
        id="BabyAI-MixedTestLocal-v0",
        entry_point="minigrid.envs.babyai:Level_MixedTestLocal",
    )


register_minigrid_envs()

try:
    import sys

    from farama_notifications import notifications

    if "minigrid" in notifications and __version__ in notifications["minigrid"]:
        print(notifications["minigrid"][__version__], file=sys.stderr)
except Exception:  # nosec
    pass
