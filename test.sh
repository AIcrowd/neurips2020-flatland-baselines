echo "===================="
echo "MARWIL"
echo "===================="
time python ./trainImitate.py -f experiments/tests/MARWIL.yaml

echo "===================="
echo "GLOBAL OBS"
echo "===================="
time python ./train.py -f experiments/tests/global_obs.yaml