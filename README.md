# senior-proj-ai

## Python Environment

### Setup a new environment

```sh
# install miniconda version
pyenv install miniconda3-4.3.30

# activate pyenv miniconda
pyenv activate miniconda3-4.3.30

# create anaconda environment
conda create -n senior-proj-ai anaconda

# activate anaconda
conda activate senior-proj-ai

# add conda-forge channel
conda config --prepend channels conda-forge 

# install requirements
conda install --file environment.yml
```

### Activate the environment

```sh
source ./pyenv.sh
```
