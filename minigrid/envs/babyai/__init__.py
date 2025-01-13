from __future__ import annotations

from minigrid.envs.babyai.goto import (
    GoTo,
    GoToDoor,
    GoToImpUnlock,
    GoToLocal,
    GoToObj,
    GoToObjDoor,
    GoToRedBall,
    GoToRedBallGrey,
    GoToRedBallNoDists,
    GoToRedBlueBall,
    GoToSeq,
)
from minigrid.envs.babyai.mixed_seq_levels import (
    Level_MixedTestLocal,
    Level_MixedTrainLocal,
)
from minigrid.envs.babyai.open import (
    Open,
    OpenDoor,
    OpenDoorsOrder,
    OpenRedDoor,
    OpenTwoDoors,
)
from minigrid.envs.babyai.other import (
    ActionObjDoor,
    FindObjS5,
    KeyCorridor,
    MoveTwoAcross,
    OneRoomS8,
)
from minigrid.envs.babyai.pickup import (
    Pickup,
    PickupAbove,
    PickupDist,
    PickupLoc,
    UnblockPickup,
)
from minigrid.envs.babyai.putnext import PutNext, PutNextLocal
from minigrid.envs.babyai.synth import (
    BossLevel,
    BossLevelNoUnlock,
    MiniBossLevel,
    Synth,
    SynthLoc,
    SynthSeq,
)
from minigrid.envs.babyai.unlock import (
    BlockedUnlockPickup,
    KeyInBox,
    Unlock,
    UnlockLocal,
    UnlockPickup,
    UnlockToUnlock,
)
