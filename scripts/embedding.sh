cp ./embedder/aic24_extract.py ./deep-person-reid/torchreid/
cd ./deep-person-reid
conda activate torchreid

for SCENE in $*
do
    echo Processing scene $SCENE
    python3 torchreid/aic24_extract.py -s $SCENE ../
done