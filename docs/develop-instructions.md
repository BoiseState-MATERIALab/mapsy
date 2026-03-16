Notes:
* MapSy was originally developed using Python 3.9, but it is currently ported and tested on 3.11. 
* Python 3.14 is not supported yet (issues with Pydantic).

Conda Environment (recommended):
1. Install miniconda and set it up with conda-forge and mamba (`conda install -n base -c conda-forge mamba`)
2. Create a dedicated conda environment for MapSy (`mamba create -n mapsy311 python=3.11`) 
3. Activate the environment (`conda activate mapsy311`)
3. Install ipykernel (for testing notebooks) and flit (to compile Mapsy)
    3.1 Upgrade pip `python -m pip install --upgrade pip`
    3.2 `python -m pip install ipykernel`
    3.3 `python -m pip install flit`


Develop Instructions:
1. Install Mapsy `flit install --symlink` or `python -m flit install --symlink` (you may need to install flit first, see above)
2. don't commit jupyter notebooks with outputs 
3. double clean jupyter notebooks with 'nb-clean clean -e notebook.ipynb'
4. tests folder is only for actual tests of mapsy functionalities
5. examples may contain actual calculations, but production calculations should not be included in the main branch
6. if you are in doubt, create a new branch and push it to the remote, everybody will be able to access it and check your stuff without you messing with the main code