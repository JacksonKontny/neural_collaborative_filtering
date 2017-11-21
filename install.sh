curl -O https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh

# Type yes to agree to license terms
bash Anaconda3-4.2.0-Linux-x86_64.sh

# activate the conda installation
source ~/.bashrc

# Create the tensorflow anaconda environment
conda-env create -f environment.yml -n ncf

# Activate the tensorflow environment
source activate ncf
