cp ./embedder/aic24_extract.py ./deep-person-reid/torchreid/
cd ./deep-person-reid
conda activate torchreid

for SCENE in $*
do
    F_SCENE=$(printf "%03d" "$SCENE")
    echo Procssing scene-$F_SCENE
    python3 torchreid/aic24_extract.py -s scene_$F_SCENE ../
done