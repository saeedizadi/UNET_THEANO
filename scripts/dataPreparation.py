import os
import tqdm
import  glob
import  numpy as np
import cv2
from PIL import Image, ImageOps
import cPickle as pickle
from eliaLib import dataRepresentation



class dataPreparation():
    def __init__(self, imgdir, gtdir):
        self.imgdir = imgdir
        self.gtdir = gtdir
        
    def represent(self, savePath, phase='train'):
        
        listFilesTrain = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(self.imgdir, phase + '_*'))]        
        data = []
        for currFile in tqdm(listFilesTrain):
            data.append(dataRepresentation.Target(os.path.join(self.imgdir, currFile + '.bmp'),
                                                       os.path.join(self.gtdir, currFile + '.bmp'),
                                                       os.path.join('' , currFile + '.mat'),
                                                       dataRepresentation.LoadState.loaded, dataRepresentation.InputType.image,
                                                       dataRepresentation.LoadState.loaded, dataRepresentation.InputType.imageGrayscale,
                                                       dataRepresentation.LoadState.unloaded, dataRepresentation.InputType.empty))

        with open(os.path.join(savePath, phase + 'Data.pickle'), 'wb') as f:
            pickle.dump(data, f)

    def resize_all(self, width=320, height=240):
        listImgFiles = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(self.imgdir, '*'))]
        for currFile in tqdm(listImgFiles):
            tt = dataRepresentation.Target(os.path.join(self.imgdir, currFile + '.bmp'),
                                           os.path.join(self.gtdir, currFile + '.bmp'),
                                           os.path.join('', currFile + '.mat'),
                                           dataRepresentation.LoadState.loaded, dataRepresentation.InputType.image,
                                           dataRepresentation.LoadState.loaded, dataRepresentation.InputType.imageGrayscale,
                                           dataRepresentation.LoadState.unloaded, dataRepresentation.InputType.empty)

            imageResized = cv2.cvtColor(cv2.resize(tt.image.getImage(), (width, height), interpolation=cv2.INTER_AREA),
                                        cv2.COLOR_RGB2BGR)
            segResized = cv2.resize(tt.saliency.getImage(), (width, height), interpolation=cv2.INTER_AREA)

            cv2.imwrite(os.path.join(self.imgdir, currFile + '.bmp'), imageResized)
            cv2.imwrite(os.path.join(self.gtdir, currFile + '.bmp'), segResized)

    def augment(self, rotation=True, HFlip=True, VFlip=True):
        listImgFiles = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(self.imgdir, '*'))]
        
        listFilesTrain = [k for k in listImgFiles if 'train' in k]
        listFilesVal = [k for k in listImgFiles if 'train' not in k]
        
        for filenames in tqdm(listFilesTrain):
            if rotation:
                for angle in [90, 180, 270]:
                    src_im = Image.open(os.path.join(self.imgdir, filenames + '.bmp'))
                    gt_im = Image.open(os.path.join(self.gtdir, filenames + '.bmp'))
                    rot_im = src_im.rotate(angle, expand=True)
                    rot_gt = gt_im.rotate(angle, expand=True)
                    rot_im.save(os.path.join(self.imgdir, filenames + '_' + str(angle) + '.bmp'))
                    rot_gt.save(os.path.join(self.gtdir, filenames + '_' + str(angle) + '.bmp'))
            if VFlip:
                vert_im = ImageOps.flip(src_im)
                vert_gt = ImageOps.flip(gt_im)
                vert_im.save(os.path.join(self.imgdir, filenames + '_vert.bmp'))
                vert_gt.save(os.path.join(self.gtdir, filenames + '_vert.bmp'))
            if HFlip:
                horz_im = ImageOps.mirror(src_im)
                horz_gt = ImageOps.mirror(gt_im)
                horz_im.save(os.path.join(self.imgdir, filenames + '_horz.bmp'))
                horz_gt.save(os.path.join(self.gtdir, filenames + '_horz.bmp'))
                
    def partition(self, fraction):
        imgFiles = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(self.imgdir, '*'))]
        numSamples = len(imgFiles)
        train_ind = np.random.choice(np.arange(0, numSamples), (1, int(np.floor(fraction * numSamples))),
                                     replace=False).squeeze()
        val_ind = np.array(np.setdiff1d(np.arange(0, numSamples), train_ind)).squeeze()

        for k in train_ind:            
            os.rename(os.path.join(self.imgdir, imgFiles[k] + '.bmp'),
                      os.path.join(self.imgdir, 'train_' + imgFiles[k] + '.bmp'))
            os.rename(os.path.join(self.gtdir, imgFiles[k] + '.bmp'),
                      os.path.join(self.gtdir, 'train_' + imgFiles[k] + '.bmp'))
        for k in val_ind:
            os.rename(os.path.join(self.imgdir, imgFiles[k] + '.bmp'),
                      os.path.join(self.imgdir, 'val_' + imgFiles[k] + '.bmp'))
            os.rename(os.path.join(self.gtdir, imgFiles[k] + '.bmp'),
                      os.path.join(self.gtdir, 'val_' + imgFiles[k] + '.bmp'))