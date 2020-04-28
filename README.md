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

## Jupyter Notebook Extension (Optional)

```sh
source ./pyenv.sh

jupyter contrib nbextension install

# autopep8
jupyter nbextension enable code_prettify/autopep8

# vim_binding
cd $PYENV_ROOT/versions/miniconda3-4.3.30/envs/senior-proj-ai/share/jupyter/nbextensions
git clone https://github.com/lambdalisue/jupyter-vim-binding vim_binding
jupyter nbextension enable vim_binding/vim_binding
```
