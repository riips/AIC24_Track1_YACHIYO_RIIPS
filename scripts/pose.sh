cp ./poser/load_tracking_result.py ./mmpose/demo/
cp ./poser/top_down_video_demo_with_track_file.py ./mmpose/demo/
cd ./mmpose
conda activate openmmlab

for SCENE in $*
do
    F_SCENE=$(printf "%03d" "$SCENE")
    echo Procssing scene-$F_SCENE
    find "../Detection/scene_$F_SCENE" -maxdepth 1 -type f -name "*.txt" | while read -r file;
    do
        CAMERA=$(basename "$file")
        number=$(echo "$CAMERA" | sed 's/camera_\([0-9]\+\).txt/\1/')
        python3 demo/top_down_video_demo_with_track_file.py ../Detection/scene_${F_SCENE}/${CAMERA} ./configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth --video-path ../Original/scene_${F_SCENE}/camera_${number}/video.mp4 --out-file ../Pose/scene_${F_SCENE}/camera_${number}/camera_${number}_out_keypoint.json
    done
done
