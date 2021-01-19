import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)
from utils.ShapeNet import ShapeNet
from utils.transform import transform
from torch_geometric.data import DataLoader
from utils.data_loader import get_ShapeNet
from models.PPFNet import get_model
from utils.transform import transform
from utils.evaluation import evaluate_PN
from utils.equal_dataset import get_equal_dataset
#from utils.plot_utils import create_figs
import torch
import cv2
from utils.explainability import *
rotated_SAVEPATH = 'code/models/saved_models/new_PPFNetmodel_rotate.pkl'
train_data,test_data = get_ShapeNet(root = 'data/ShapeNet',split = 0.8,transformation=transform(points=256))
batches = DataLoader(train_data[0:2], batch_size=1,shuffle = True)
model = get_model(train_data.num_classes,hidden_layers = 16)
print(model)


grad_cam = GradCam(model=model, feature_module=model.conv3, \
                       target_layer_names=["2"], use_cuda=False)

#img = cv2.imread(args.image_path, 1)
#img = np.float32(img) / 255
# Opencv loads as BGR:
#img = img[:, :, ::-1]
for data in batches:
	input_img = data
	break
print(input_img)

#print(input_img.pos)
# If None, returns the map for the highest scoring category.
# Otherwise, targets the requested category.
target_category = None
grayscale_cam = grad_cam(input_img, target_category)

grayscale_cam = cv2.resize(grayscale_cam, (img.shape[1], img.shape[0]))
cam = show_cam_on_image(img, grayscale_cam)

#gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
#gb = gb_model(input_img, target_category=target_category)
#gb = gb.transpose((1, 2, 0))

#cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
#cam_gb = deprocess_image(cam_mask*gb)
#gb = deprocess_image(gb)

cv2.imwrite("cam.jpg", cam)
#cv2.imwrite('gb.jpg', gb)
#cv2.imwrite('cam_gb.jpg', cam_gb)