SCENE=$*

#cd tracking

for SCENE in $*
do
    echo Processing scene-$SCENE
    python tracking/src/run.py -s $SCENE
done