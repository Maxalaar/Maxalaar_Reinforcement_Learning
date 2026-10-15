PROJECTNAME=Maxalaar_Reinforcement_Learning

mkdir -p ~/packages/$PROJECTNAME

module load aidl/pytorch/2.2.0-cuda12.1
export PYTHONUSERBASE=~/packages/$PROJECTNAME
export PATH=$PATH:~/packages/$PROJECTNAME/

pip install --upgrade pip

# Dependencies are listed in pyproject.toml; add the environments needed on the cluster as extras.
pip install --user --no-cache-dir -e ".[box2d,mujoco,atari,flappy-bird,tetris]"
# Ray optimizer-restore patch (see README).
python3 scripts/install_ray_patch.py
