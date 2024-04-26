SCENE=$*

#cd tracking

for SCENE in $*
do
    echo Procssing scene-$SCENE
    python tracking/infer.py -s $SCENE
done