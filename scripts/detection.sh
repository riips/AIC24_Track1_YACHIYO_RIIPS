cp ./detector/aic24_get_detection.py ./BoT-SORT/tools/
cd ./BoT-SORT
conda activate botsort_env

for SCENE in $*
do
    F_SCENE=$(printf "%03d" "$SCENE")
    echo Procssing scene-$F_SCENE
    python3 tools/aic24_get_detection.py -s scene_$F_SCENE ../
done