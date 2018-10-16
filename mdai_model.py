import os
import sys
import random
import math
import numpy as np
import cv2
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import json
import pydicom
from imgaug import augmenters as iaa
from tqdm import tqdm
import pandas as pd
import glob

DATA_DIR = '/data/krf/dataset'
MODEL_DIR = '/data/krf/model/rsna'

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

train_dicom_dir = os.path.join(DATA_DIR,'stage_1_train_images')
test_dicom_dir = os.path.join(DATA_DIR,'stage_1_test_images')

def get_dicom_fps(dicom_dir):
	dicom_fps = glob.glob(dicom_dir+'/'+'*.dcm')
	return list(set(dicom_fps))

def parse_dataset(dicom_dir, anns):
	image_fps = get_dicom_fps(dicom_dir)
	image_annotations = {fp: [] for fp in image_fps}
	for index, row in anns.iterrows():
		fp = os.path.join(dicom_dir, row['patientId']+'.dcm')
		image_annotations[fp].append(row)
	return image_fps, image_annotations

class DetectorConfig(Config):
	NAME ='pneumonia'
	GPU_COUNT = 1
	IMAGES_PER_GPU = 8

	BACKBONE = 'resnet50'

	NUM_CLASSES = 2

	IMAGE_MIN_DIM = 512
	IMAGE_MAX_DIM = 512

	RPN_ANCHOR_SCALES = (32,64,128,256)

	TRAIN_ROIS_PER_IMAGE = 32

	MAX_GT_INSTANCES = 5

	DETECTION_MAX_INSTANCES = 3
	DETECTION_MIN_CONFIDENCE = 0.4
	DETECTION_NMS_THRESHOLD = 0.1

	RPN_TRAIN_ANCHORS_PER_IMAGE = 16
	TOP_DOWN_PYRAMID_SIZE = 32
	STEPS_PER_EPOCH = 1024

config = DetectorConfig()
config.display()


class DetectorDataset(utils.Dataset):
	
	def __init__(self, image_fps, image_annotations, orig_height, orig_width):
		super().__init__(self)

		self.add_class('pneumonia', 1, 'Lung Opacity')
		for i, fp in enumerate(image_fps):
			annotations = image_annotations[fp]
			self.add_image('pneumonia', image_id = i, path=fp,annotations = annotations,orig_height = orig_height,orig_width=orig_width)
	
	def image_reference(self,image_id):
		info = self.image_info[image_id]
		return info['path']

	def load_image(self,image_id):
		info = self.image_info[image_id]
		fp = info['path']
		ds = pydicom.read_file(fp)
		image = ds.pixel_array

		if len(image.shape)!=3 or image.shape[2]!=3:
			image = np.stack((image,) * 3, -1)
		return image

	def load_mask(self, image_id):
		info = self.image_info[image_id]
		annotations = info['annotations']
		count = len(annotations)
		if count == 0:
			mask = np.zeros((info['orig_height'], info['orig_width'],1), dtype=np.uint8)
			class_ids = np.zeros((1,),dtype=np.int32)
		else:
			mask = np.zeros((info['orig_height'], info['orig_width'],count),dtype=np.uint8)
			class_ids = np.zeros((count,),dtype=np.int32)
			for i, a in enumerate(annotations):
				if a['Target'] == 1:
					x = int(a['x'])
					y = int(a['y'])
					w = int(a['width'])
					h = int(a['height'])
					mask_instance = mask[:,:,i].copy()
					cv2.rectangle(mask_instance, (x,y),(x+w,y+h),255,-1)
					mask[:,:,i] = mask_instance
					class_ids[i] = 1
		return mask.astype(np.bool), class_ids.astype(np.int32)

anns = pd.read_csv(os.path.join(DATA_DIR,'stage_1_train_labels.csv'))
#print(os.path.join(DATA_DIR,'stage_1_train_labels.csv'))
print(anns.head(6))

image_fps, image_annotations = parse_dataset(train_dicom_dir,anns=anns)
ds = pydicom.read_file(image_fps[0])
image = ds.pixel_array

print(ds)

ORIG_SIZE = 1024
#TRAINING NUM
image_fps_list = list(image_fps)

sorted(image_fps_list)
random.seed(42)
random.shuffle(image_fps_list)

validation_split = 0.1
split_index = int((1 - validation_split) * len(image_fps_list))

image_fps_train = image_fps_list[:split_index]
image_fps_val = image_fps_list[split_index:]

print(len(image_fps_train),len(image_fps_val))


dataset_train = DetectorDataset(image_fps_train, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_train.prepare()

test_fp = random.choice(image_fps_train)
print(image_annotations[test_fp])


dataset_val = DetectorDataset(image_fps_val, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_val.prepare()


#load and display random samples

image_id = random.choice(dataset_train.image_ids)
image_fp = dataset_train.image_reference(image_id)
image = dataset_train.load_image(image_id)
mask, class_ids = dataset_train.load_mask(image_id)

print(image.shape)

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(image[:,:,0],cmap='gray')
plt.axis('off')
#plt.savefig(MODEL_DIR+str(image_id)+'.png')

plt.subplot(1,2,2)
masked = np.zeros(image.shape[:2])
for i in range(mask.shape[2]):
	masked += image[:,:,0] * mask[:,:,i]
plt.imshow(masked,cmap='gray')
plt.axis('off')

plt.savefig(MODEL_DIR+'/'+str(image_id)+'.png')
print(image_fp)
print(class_ids)



model = modellib.MaskRCNN(mode='training', config=config, model_dir = MODEL_DIR)

#Image augmentation
augmentation = iaa.SomeOf((0,1),[
	iaa.Fliplr(0.5),
	iaa.Affine(
		scale={"x":(0.8,1.2),"y":(0.8,1.2)},
		translate_percent = {"x":(-0.2,0.2),"y":(-0.2,0.2)},
		rotate=(-25,25),
		shear=(-8,8)
	),
	iaa.Multiply((0.9,1.1))
])


#train
#NUM_EPOCHS = 20
#
#import warnings
#warnings.filterwarnings("ignore")
#model.train(dataset_train, dataset_val,
#	learning_rate=config.LEARNING_RATE,
#	epochs=NUM_EPOCHS,
#	layers='all',
#	augmentation=augmentation)

#select trained model
dir_names = next(os.walk(model.model_dir))[1]
key = config.NAME.lower()
dir_names = filter(lambda f: f.startswith(key),dir_names)
dir_names = sorted(dir_names)

if not dir_names:
	import errno
	raise FileNotFoundError(
		errno.ENOENT,
		"Could not find model directory under {}".format(self.model_dir))

fps = []
for d in dir_names:
	dir_name = os.path.join(model.model_dir,d)
	#Find the last checkpoint
	checkpoints = next(os.walk(dir_name))[2]
	checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
	checkpoints = sorted(checkpoints)
	if not checkpoints:
		print('No weight files in {}'.format(dir_name))
	else:
		checkpoint = os.path.join(dir_name,checkpoints[-1])
		fps.append(checkpoint)

model_path = sorted(fps)[-1]
print('Found model {}'.format(model_path))

class InferenceConfig(DetectorConfig):
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
inference_config = InferenceConfig()

#Recreate the model in inference mode
model = modellib.MaskRCNN(mode='inference',config=inference_config,model_dir=MODEL_DIR)

#Load trained weights(fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ",model_path)
model.load_weights(model_path,by_name=True)


#Set color for class
def get_colors_for_class_ids(class_ids):
	colors=[]
	for class_id in class_ids:
		if class_id == 1:
			colors.append((.941, .204, .204))
	return colors

#Show few example of ground truth vs. predictions on the validation dataset
dataset = dataset_val
fig = plt.figure(figsize=(10,30))

for i in range(4):
	image_id = random.choice(dataset.image_ids)
	original_image, image_meta, gt_class_id, gt_bbox, gt_mask =modellib.load_image_gt(dataset_val,inference_config,image_id,use_mini_mask = False)
	plt.subplot(6,2,2*i+1)
	visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
		dataset.class_names,
		colors=get_colors_for_class_ids(gt_class_id),ax=fig.axes[-1])
	
	plt.subplot(6,2,2*i+2)
	results = model.detect([original_image])
	r = results[0]
	visualize.display_instances(original_image, r['rois'],r['masks'],r['class_ids'],
		dataset.class_names,r['scores'],
		colors=get_colors_for_class_ids(r['class_ids']),ax=fig.axes[-1])
plt.savefig(MODEL_DIR+"/example.png")

#Get filenames of test dataset DICOM images
test_image_fps = get_dicom_fps(test_dicom_dir)

#Make predictions on test images, write out sample submission
def predict(image_fps, filepath='sample_submission.csv',min_conf=0.98):
	#assume square image

	with open(filepath,'w') as file:
		file.write("patientId,PredictionString\n")
		for image_id in tqdm(image_fps):
			ds = pydicom.read_file(image_id)
			image = ds.pixel_array

			#if grayscale, convert to rgb for consistency
			if len(image.shape)!=3 or image.shape[2]!=3:
				image=np.stack((image,)*3,-1)
			patient_id = os.path.splitext(os.path.basename(image_id))[0]
			
			results = model.detect([image])
			r = results[0]
			
			out_str = ""
			out_str += patient_id
			assert( len(r['rois']) == len(r['class_ids']) == len(r['scores']) )
			if len(r['rois']) == 0:
				out_str += ", "
				pass
			else:
				num_instances = len(r['rois'])
				out_str += ","
				for i in range(num_instances):
					if r['scores'][i] > min_conf:
						out_str += ' '
						out_str += str(round(r['scores'][i],2))
						out_str += ' '

						#x1,y1,width,height
						x1 = r['rois'][i][1]
						y1 = r['rois'][i][0]
						width = r['rois'][i][3] - x1
						height = r['rois'][i][2] -y1
						bboxes_str = "{} {} {} {}".format(x1,y1,width,height)
						out_str += bboxes_str
			file.write(out_str+"\n")

#predict only the first 50 entries
sample_submission_fp = 'sample_submission.csv'
predict(test_image_fps, filepath=sample_submission_fp)

output = pd.read_csv(sample_submission_fp, names=['id','pred_string'])
print(output.head(50))



	
