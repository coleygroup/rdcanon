# rdcanon - SMARTS and Reaction SMARTS Canonicalization

## Overview
rdcanon is a package designed for canonicalizing SMARTS and Reaction SMARTS templates. It reorders SMARTS to optimize querying speed. This optimization is invariant of atom mapping.

## Installation
### Prerequisites
- Ensure you have rdkit installed (version > 2023.9.2).

### Steps
1. Create or activate a virtual environment.
2. Clone the repository.
3. Install the package with the command:
>pip install -e rdcanon2024

Note: rdkit will be installed automatically if it's not already present in the environment.

## Usage
### Sanitizing Individual SMARTS
To sanitize individual SMARTS:
```python
from rdcanon import canon_smarts 

test_smarts = [
 "[$([NX3H,NX4H2+]),$([NX3](C)(C)(C))]1[CX4H]([CH2][CH2][CH2]1)[CX3](=[OX1])[OX2H,OX1-,N]",
 "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([*])[CX3](=[OX1])[OX2H,OX1-,N]",
 "[CX3](=O)[OX1H0-,OX2H1]",
 "[CX3](=[OX1])[OX2][CX3](=[OX1])",
 "[N&H2&+0:4]-[C&H1&+0:2](-[C&H2&+0:8])-[O&H1&+0:3]"
]

# The second parameter is optional and flags whether atom mapping should be returned (defaults to False)
for smarts in test_smarts:
 print(smarts, canon_smarts(smarts), canon_smarts(smarts, True))
```

### Sanitizing Reaction SMARTS
For sanitizing reaction SMARTS:
```python
from rdcanon import canon_reaction_smarts
```

