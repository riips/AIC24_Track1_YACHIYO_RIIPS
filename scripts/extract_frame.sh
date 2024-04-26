conda activate botsort_env

for SCENE in $*
do
    F_SCENE=$(printf "%03d" "$SCENE")
    echo Procssing scene-$F_SCENE
    python3 tools/extract_frame.py -s scene_$F_SCENE ./
done