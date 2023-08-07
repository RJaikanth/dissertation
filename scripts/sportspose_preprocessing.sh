current_dir=$(pwd)
export PYTHONPATH=$PYTHONPATH:$current_dir

python3 src/preprocessing/sports_pose.py

#cd dataset/SportsPose/final_ds
#kaggle datasets create -p ./ --dir-mode zip
