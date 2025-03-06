from typing import Literal

# yapf: disable

# literal type aliases

SystemType = Literal[
    'ions',
    'electrons',
    'full',
]

FileFormat = Literal[
    'xyz+',
    'ase',
    'cube',
]

Units = Literal[
    'bohr',
    'angstrom',
    'alat',
]

ContactSpaceMode = Literal[
    'ionic',
    'electronic',
    'system',
]

RadiusMode = Literal[
    'pauling',
    'bondi',
    'uff',
    'muff',
]
