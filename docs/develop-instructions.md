1. flit install --symlink (you may need to install flit first)
2. don't commit jupyter notebooks with outputs 
3. double clean jupyter notebooks with 'nb-clean clean -e notebook.ipynb'
4. tests folder is only for actual tests of mapsy functionalities
5. examples may contain actual calculations, but production calculations should not be included in the main branch
6. if you are in doubt, create a new branch and push it to the remote, everybody will be able to access it and check your stuff without you messing with the main code