'''
extract ReID features from testing data.
'''
import os
import argparse
import os.path as osp
import numpy as np
import torch
import time
import torchvision.transforms as T
from PIL import Image
import sys
from utils import FeatureExtractor
import torchreid
import json

def make_parser():
    parser = argparse.ArgumentParser("reid")
    parser.add_argument("root_path", type=str, default=None)
    parser.add_argument("-s", "--scene", type=str, default=None)
    return parser

if __name__ == "__main__":

    args = make_parser().parse_args()
    data_root = args.root_path
    scene = args.scene

    sys.path.append(data_root+'/deep-person-reid')

    img_dir = os.path.join(data_root,'Original')
    det_dir = os.path.join(data_root,'Detection')
    out_dir = os.path.join(data_root,'EmbedFeature')

    models = {
              'osnet_x1_0':data_root+'/deep-person-reid/checkpoints/osnet_ms_m_c.pth.tar'
             }
    
    
    model_names = ['osnet_x1_0']
    

    val_transforms = T.Compose([
        T.Resize([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    for model_idx,name in enumerate(models):
        
        model_p = models[name]
        model_name = model_names[model_idx]

        print('Using model {}'.format(name))

        extractor = FeatureExtractor(
            model_name=model_name,
            model_path=model_p,
            device='cuda'
        )   

        for file in os.listdir(os.path.join(det_dir,scene)):
            base, ext = os.path.splitext(file)
            if ext == '.txt':
                print('processing file {}{}'.format(base,ext))
                det_path = os.path.join(det_dir,scene,'{}.txt'.format(base))
                json_path = os.path.join(det_dir,scene,'{}.json'.format(base))
                dets = np.genfromtxt(det_path,dtype=str,delimiter=',')
                with open(json_path) as f:
                    jf = json.load(f)
                cur_frame = 0
                u_num = 0
                emb = np.array([None]*len(dets))
                start = time.time()
                print('processing scene {} cam {} with {} detections'.format(scene,base,len(dets)))
                for idx,(cam,frame,_,x1,y1,x2,y2,conf) in enumerate(dets):
                    u_num += 1
                    x1,y1,x2,y2 = map(float,[x1,y1,x2,y2])
                    if idx%1000 == 0:
                        if idx !=0:
                            end = time.time()
                            print('processing time :',end-start)
                        start = time.time()
                        print('process {}/{}'.format(idx,len(dets)))
                    if cur_frame != int(frame):
                        cur_frame = int(frame)
                    if not os.path.isdir(osp.join(out_dir,scene,cam)):
                        os.makedirs(osp.join(out_dir,scene,cam))
                    save_fn = os.path.join(out_dir,scene,cam,'feature_{}_{}_{}_{}_{}_{}_{}.npy'.format(cur_frame,u_num,str(int(x1)),str(int(x2)),str(int(y1)),str(int(y2)),str(conf).replace(".","")))
                    jf[str(idx).zfill(8)]['NpyPath'] = os.path.join(scene,cam,'feature_{}_{}_{}_{}_{}_{}_{}.npy'.format(cur_frame,u_num,str(int(x1)),str(int(x2)),str(int(y1)),str(int(y2)),str(conf).replace(".","")))
                    img_path = os.path.join(img_dir,scene,cam,'Frame',frame.zfill(6)+'.jpg')
                    img = Image.open(img_path)
        
                    img_crop = img.crop((x1,y1,x2,y2))
                    img_crop = val_transforms(img_crop.convert('RGB')).unsqueeze(0)
                    feature = extractor(img_crop).cpu().detach().numpy()[0]
    
                    np.save(save_fn,feature)
                end = time.time()
                print('processing time :',end-start)
                start = time.time()
                print('process {}/{}'.format(idx+1,len(dets)))
                with open(json_path, 'w') as f:
                    json.dump(jf, f, ensure_ascii=False)
