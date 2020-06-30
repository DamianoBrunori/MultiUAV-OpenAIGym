#!/bin/sh
xhost +"local:docker@"
cd custom_gym/
pip3 install -e . 

echo "
 ____   ____ _____ _   _ 
/ ___| / ___| ____| \ | |
\___ \| |   |  _| |  \| |
 ___) | |___| |___| |\  |
|____/ \____|_____|_| \_|"
                         
python3 scenario_objects.py

echo "
 ____  _     ___ _____ 
|  _ \| |   / _ \_   _|
| |_) | |  | | | || |  
|  __/| |__| |_| || |  
|_|   |_____\___/ |_|  "

python3 plotting.py

echo "
 _____ _   ___     __
| ____| \ | \ \   / /
|  _| |  \| |\ \ / / 
| |___| |\  | \ V /  
|_____|_| \_|  \_/   "

python3 env_wrapper.py
