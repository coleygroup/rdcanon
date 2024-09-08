# rdcanon - SMARTS and Reaction SMARTS Canonicalization

## Overview
rdcanon is a package designed for canonicalizing SMARTS and Reaction SMARTS templates. It reorders SMARTS to optimize querying speed. This optimization is invariant of atom mapping.

## Installation
### Prerequisites
- Ensure you have rdkit installed (version > 2023.9.2).
- The following packages will be installed:
        'rdkit > 2023.09.1',
        'matplotlib',
        'lark',
        'numpy',
        'networkx',
        'scikit-learn', (optional, for kde generation)
        'ipykernel',
        'pandas',
        'openpyxl'

### Steps
1. Create or activate a virtual environment.
2. Clone the repository.
3. Install the package with the command:
>pip install -e rdcanon


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

### Unit Testing
To run all unit tests:
>python rdcanon_tests.py

note: currently, "TestRecursive.test_validate_recursive_against_database" is expected to fail at: "[#1][C&X3]([#1,#6])=[O&X1] [#1,#6][C&X3;H1,H2]=[O&X1]". This is due to a strange edge case with merging query Hs, explicit hydrogens, and explicit/implicit connections.


### Current Limitations
No consolidation or expansion of atomic queries is performed automatically, but a mechanism is provided to allow the user to systematically replace canonicalized atomic queries with an input dictionary (e.g., {"[O;H1]": "[O;H1;+0]"} would replace the canonicalized variant of [O;H1] with the canonicalized variant of [O;H1;+0]). 

Replacement dictionaries should be processed first using 
```python
canon_repl_dict = gen_canon_repl_dict(repl_dict)
```

before passing as an argument into canon_smarts.

Chirality or directionality beyond tetrahedral centers and cis/trans isomerism is not currently supported.

### Manuscript Figures and Tests
All data can be found in the manuscripts/data directory.

To create the bar charts of Figure 1, use the notebook within the manuscript directory named "prim_frequencies.ipynb". 

To run the subgraph isomorphism experiments of Figure 3, use the notebook within the manuscript directory named "generate_plots_substruct_match.ipynb". 

To run the template application experiments of Figure 4, use the notebook within the manuscript directory named "gen_plots_run_reactants_20240104.ipynb". 

To run the retrosynthetic analysis experiments of Figure 5, use the notebook within the manuscript directory named "gen_retrosim_plots_20240105.ipynb". 


### RDCanon Files
The main workflow consists of two files, main.py and token_parser.py. The main.py file calls token_parser.py to parse and score atomic queries.

The files askcos_prims.py, drugbank_prims_with_nots.py, np_prims.py, and pubchem_prims.py are 4 query primitive frequency dictonaries, which are used for embedding leaf nodes in query trees.

The rdcanon_tests.py file contains all of the test cases using the abseil interface.

Finally, utils.py contains some helper functions for testing and plotting.
