#!/usr/bin/env python
# coding: utf-8

# In[4]:
"""

#load and vis a few seqmented images in the images_CNN folder
import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import Image, ImageSequence
img_list = glob.glob('images_CNN/F?_im.TIF')+glob.glob('images_CNN/F??_im.TIF')
img_list = sorted(img_list)
mask_list = glob.glob('images_CNN/*mask.h5')
mask_list = sorted(mask_list)
fluo_gfp = glob.glob('images_CNN/*GFP_im.TIF')
fluo_gfp = sorted(fluo_gfp)
fluo_rfp = glob.glob('images_CNN/*RFP_im.TIF')
fluo_rfp = sorted(fluo_rfp)
print( len(img_list), len(mask_list) , len(fluo_gfp), len(fluo_rfp  )   )
dataset = {}
for i in range(len(img_list)):
    dataset[i] = { 'img': img_list[i], 'mask': mask_list[i], 'gfp': fluo_gfp[i], 'rfp': fluo_rfp[i]  }
print(dataset)


# In[9]:


from PIL import Image, ImageEnhance
def adjust_brightness_contrast(input_image_path, output_image_path, brightness=3, contrast=3):
    # Open the image
    image = Image.open(input_image_path)
    
    # Adjust brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    
    # Adjust contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    
    # Save the modified image
    image.save(output_image_path)


# In[10]:


import skimage.measure as measure
import copy
def output_contours( m , cl , verbose = False):

    c = []
    for val in list(np.unique(m)):
        sub = copy.deepcopy(m)
        sub[sub!=val]= 0
        c+= measure.find_contours(sub, .9)
    contours = c
    
    if verbose == True:
        plt.imshow(m)
        plt.title( 'contours ' + str(cl) ) 
        for n, contour in enumerate(contours):
            plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
        plt.show()

    #output contours of each mask to file
    #divide x and y coordinates by total image size
    #to get values between 0 and 1  
    #<class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>
    lines = []
    
    for c in contours:
        coords = []
        for i in range(0,c.shape[0]):
            coords.append( (float(c[i][1]) / m.shape[0]) )
            coords.append( (float(c[i][0]) / m.shape[1]) )
        line = str(cl) + ' ' + ' '.join([str(c) for c in coords]) + '\n'
        lines.append(line)
    return lines

def split_mask(mask, crop = 1024):
    #custom encoding with 3 classes
    mask = mask[0:crop, 0:crop]
    mask1 = copy.deepcopy(mask)
    mask1[mask1 > 1000] =  0
    
    mask2 = copy.deepcopy(mask)
    mask2[mask2 < 1000 ] = 0
    mask2[mask2 > 2000 ] = 0
    
    mask3 = copy.deepcopy(mask)
    mask3[ (mask3 < 2000) ] =  0
    
    return mask1, mask2, mask3

def mask2contourfile( mask , outputfile , verbose = False):
    if type(mask) == list:
        m1,m2,m3 = mask
    else:
        m1, m2, m3 = split_mask(mask)
    lines = output_contours(m1, 0 , verbose = verbose)
    lines += output_contours(m2, 1, verbose = verbose)
    lines += output_contours(m3, 2, verbose = verbose)
    
    with open(outputfile, 'w') as f:
        for l in lines:
            f.write(l)
    return  outputfile


# In[11]:


#clean finaldataset folder
import shutil
overwite = True
if overwite:
    try:
        shutil.rmtree('./datasets/')
    except:
        pass
    os.mkdir('./datasets/')
    os.mkdir('./datasets/train')
    os.mkdir('./datasets/train/images/')
    os.mkdir('./datasets/train/labels/')
    
    os.mkdir('./datasets/test')
    os.mkdir('./datasets/test/images/')
    os.mkdir('./datasets/test/labels/')
    
    os.mkdir('./datasets/val/')
    os.mkdir('./datasets/val/images')
    os.mkdir('./datasets/val/labels')


# In[15]:


import pickle
with open('scalers.pkl' , 'rb') as scalerdump:
    scalers = pickle.loads( scalerdump.read())
print(scalers)


# In[16]:


#stack equivalent frames together from img, gfp and rfp and transform them to a jpg image
import cv2
import numpy as np
import os
import tqdm

verbose = False
crop = 1024



def yield_frames(img,crop=1024 , verbose = False ,scaler = True):
    for i, page in enumerate(ImageSequence.Iterator(img)):
        if verbose == True:
            plt.imshow(np.array(page))
            plt.show()
        if crop is not None:
            page = np.array(page)[0:crop, 0:crop]
        if scaler==True:
            page = (page - page.min()) / (page.max() - page.min()) * 255
        yield page

count = 0

for sample in dataset:
    maskfile = dataset[sample]['mask']
    maskh5 = h5py.File(maskfile, 'r')
    for group in maskh5.keys():
        for frame in maskh5[group]:
            mask = np.array( maskh5[group][frame] ,  dtype = np.uint16 )
            
            if np.sum(mask) > 0 :
                mask =  mask[0:crop, 0:crop]
                dataset[sample]['maskmatrix'] = mask
                print( group, frame)
                print(np.unique(mask) )
                if verbose == True:
                    plt.imshow(mask)
                    plt.show()
                converted = mask2contourfile(mask, maskfile +'converted.txt' , verbose = verbose)
                dataset[sample]['mask_poly'] = converted
                break

    print('loading img')
    img = Image.open(dataset[sample]['img'])
    img = [frame for frame in yield_frames(img,scaler = True , verbose=verbose)]
    print('loading gfp')
    
    gfp = Image.open(dataset[sample]['gfp'])
    gfp = [frame for frame in yield_frames(gfp,scaler = True,verbose=verbose)]
    
    print('loading rfp')
    rfp = Image.open(dataset[sample]['rfp'])
    rfp = [frame for frame in yield_frames(rfp,scaler = True,verbose=verbose)]
    dataset[sample]['stack']= []
    dataset[sample]['stack_mat'] ={}
    #stack the frames together
    for i in tqdm.tqdm(range(len(img))):
        im = np.stack([img[i], gfp[i], rfp[i]], axis=-1)
        dataset[sample]['stack_mat'][i] = im

        #im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite('./datasets/train/images/img_'+str(count)+'.png', im )
        #adjust_brightness_contrast('./datasets/train/images/img_'+str(count)+'.png','./datasets/train/images/img_'+str(count)+'.png')

        dataset[sample]['stack'].append('./datasets/train/images/img_'+str(count)+'.png')
        
        #save the correct mask file
        shutil.copyfile(dataset[sample]['mask_poly'], './datasets/train/labels/img_'+str(count)+'.txt' )
        count += 1


# In[17]:


check_example = True
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

if check_example == True:
    print('loading img' )
    img = Image.open(dataset[0]['img'])
    img = [frame for frame in yield_frames(img,verbose=False)]
    print('loading gfp')

    gfp = Image.open(dataset[0]['gfp'])
    gfp = [frame for frame in yield_frames(gfp,verbose=False)]
    
    print('loading rfp')
    rfp = Image.open(dataset[0]['rfp'])
    rfp = [frame for frame in yield_frames(rfp,verbose=False)]

    ax = plt.imshow( img[0] , cmap = 'Greys')
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    plt.show()
    
    ax = plt.imshow( gfp[0] , cmap = 'Greens')
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    plt.show()
    
    ax = plt.imshow( rfp[0] , cmap = 'Reds')
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    plt.show()
    
    
    ax = plt.imshow( dataset[0]['maskmatrix'] )
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    plt.show()
    
    


# In[18]:


from sklearn.preprocessing import RobustScaler
import numpy as np
import pickle



# Initialize the RobustScaler
scaler_bf = RobustScaler()

images_bf = np.stack([ dataset[sample]['stack_mat'][frame] for sample in dataset for frame in dataset[sample]['stack_mat'] ])
print('stacks' , images_bf.shape)

image_stack_bf = images_bf.reshape(-1, images_bf.shape[0])

print( 'fitting scalers ' ) 
scaler_bf.fit(image_stack_bf)

scalers = [scaler_bf]
print('scalers', scalers)
with open('scalers.pkl' , 'wb') as scalerdump:
    scalerdump.write(pickle.dumps(scalers))


# In[19]:


#apply data augmentation
#image augmentation for training

import cv2
import random
import numpy as np

def random_rotation(image, masks, angle_range):
    angle = random.uniform(-angle_range, angle_range)
    image =  Image.fromarray(  np.array( np.ones(image.shape)*256 - Image.fromarray(image)).astype(np.uint8) )
    image = image.rotate(angle)
    masks = [ Image.fromarray(m) for m in masks ]
    masks = [ m.rotate(angle) for m in masks ]
    return np.array(image), masks

def random_flip(image, masks):
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
        masks = [ cv2.flip(np.array(m), 1) for m in masks]
    return image, masks

def random_augmentation(image, masks, angle_range, crop_size):
    
    image, masks = random_rotation(image, masks, angle_range)
    
    image, masks = random_flip(image, masks)
   
    
    #image, mask = random_crop(image, mask, crop_size)
    return image, masks

#resize to original size
def resize(image, mask, size):
    image = cv2.resize(image, size)
    mask = cv2.resize(mask, size)
    return image, mask

#apply augmentation and then resize to original size
def augment_and_resize(image, masks, angle_range=180, crop_size=900, size=[1024,1024]):
    image = image.astype(np.uint8)
    image, masks = random_augmentation(image, masks, angle_range, crop_size)
    masks = [np.array(m) for m in masks ]
    #image, masks = resize(image, masks, size)
    return image, masks


# In[20]:


augmented_folder = 'augmented/'


# In[21]:


import shutil
overwite = True
if overwite:
    try:
        shutil.rmtree(augmented_folder)
    except:
        pass
    os.mkdir(augmented_folder)
    os.mkdir(augmented_folder+'train')
    os.mkdir(augmented_folder+'train/images/')
    os.mkdir(augmented_folder+'train/labels/')
    os.mkdir(augmented_folder +'test')
    os.mkdir(augmented_folder+'test/images/')
    os.mkdir(augmented_folder + 'test/labels/')
    os.mkdir(augmented_folder +'val/')
    os.mkdir(augmented_folder +'val/images')
    os.mkdir(augmented_folder + 'val/labels')


# In[ ]:


verbose = False
import tqdm 
x_augment = 10
augmented = {}

count = 0
for sample in tqdm.tqdm(dataset):
    augmented[sample] ={}
    for frame in dataset[sample]['stack_mat']:
        image = dataset[sample]['stack_mat'][frame]
        mask = dataset[sample]['maskmatrix']
        masks = split_mask(mask)
        for i in range(x_augment):
            img , masks = augment_and_resize(image,masks)
            
            for m in masks:
                m[m>0] = 1
                
            if count < 10 and verbose == True:
                plt.imshow( img[:,:,0] , cmap = 'Greys')
                plt.colorbar( location = 'left')
                plt.show()
            augmented[sample]['mask'] = mask
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(augmented_folder+'train/images/img_'+str(count)+'.png', im )
            augmented[sample]['img'] = augmented_folder+'train/images/img_'+str(count)+'.png'
            m1,m2,m3 = masks
            lines = output_contours(m1, 0 , verbose = verbose)
            lines += output_contours(m2, 1, verbose = verbose)
            lines += output_contours(m3, 2, verbose = verbose)
            with open(augmented_folder + 'train/labels/img_'+ str(count)+ '.txt', 'w') as f:
                for l in lines:
                    f.write(l)
            augmented[sample]['mask_poly'] = augmented_folder  +'train/labels/img_'+ str(count)+ '.txt'
            #add polygons
            count+=1
            


# In[1]:


#move a fraction of the training data and corresponding labels to val
import random
import shutil
import os

datasetdir = 'datasets/'#augmented_folder

files = os.listdir(datasetdir +'train/images/')
print(files[0:100], '...')
random.shuffle(files)
val_files = files[:int(len(files)*.1)]
for f in val_files:
    shutil.move(datasetdir + 'train/images/'+f, datasetdir+'val/images/'+f)
    shutil.move(datasetdir +'train/labels/'+f.replace('.png', '.txt'), datasetdir + 'val/labels/'+f.replace('.png', '.txt'))

    
files = os.listdir(datasetdir+'train/images/')
random.shuffle(files)

test_files = files[:int(len(files)*.1)]
for f in test_files:
    shutil.move(datasetdir + 'train/images/'+f, datasetdir + 'test/images/'+f)
    shutil.move(datasetdir + 'train/labels/'+f.replace('.png', '.txt'), datasetdir + 'test/labels/'+f.replace('.png', '.txt'))



# In[2]:
"""

#create traininging yaml file for the dataset
outyaml = """
train: train
val: val
test: test

names: 
    0: f
    1: h
    2: l
    
"""

with open('./dataset.yaml', 'w') as f:
    f.write(outyaml)

hyp = {}
hyp['lr0']= 0.0001 # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
hyp['lrf']= 0.00001 # final learning rate (lr0 * lrf)
hyp['momentum']= 0.05 # SGD momentum/Adam beta1
hyp['weight_decay']= 0.00001 # optimizer weight decay 5e-4
hyp['warmup_epochs']= 3.0 # warmup epochs (fractions ok)
hyp['warmup_momentum']= 0.8 # warmup initial momentum
hyp['warmup_bias_lr']= 0.01 # warmup initial bias lr
hyp['box']= 5 # box loss gain
hyp['cls']= 20 # cls loss gain (scale with pixels)
hyp['dfl']= .5 # dfl loss gain
hyp['pose']= 0 # pose loss gain
hyp['kobj']= 0 # keypoint obj loss gain
hyp['label_smoothing']= 0.0 # label smoothing (fraction)
hyp['nbs']= 64 # nominal batch size
hyp['hsv_h']= 0.01 # image HSV-Hue augmentation (fraction)
hyp['hsv_s']= 0.01 # image HSV-Saturation augmentation (fraction)
hyp['hsv_v']= 0.01 # image HSV-Value augmentation (fraction)
hyp['degrees']= 90.0 # image rotation (+/- deg)
hyp['translate']= 0.0 # image translation (+/- fraction)
hyp['scale']= 0.1 # image scale (+/- gain)
hyp['shear']= 0.0 # image shear (+/- deg)
hyp['perspective']= 0.0 # image perspective (+/- fraction), range 0-0.001
hyp['flipud']= 0.5 # image flip up-down (probability)
hyp['fliplr']= 0.5 # image flip left-right (probability)
hyp['mosaic']= 0.0 # image mosaic (probability)
hyp['mixup']= 0.0 # image mixup (probability)
hyp['copy_paste']= 0.0 # segment copy-paste (probability)

from ultralytics import YOLO
modelpath = 'yolov8l-seg_yfusion.pt'
import os

overwrite = False
if os.path.exists(modelpath) and overwrite == False:
    model = YOLO(modelpath)
else:
    #train the model
    model = YOLO("yolov8l-seg.pt")


# In[ ]:


train = True
if train == True:
    results = model.train(
            batch=1,
            device=0,
            data='./dataset.yaml',
            epochs=5000,
            imgsz=1024,
            ** hyp
        )


# In[ ]:


#save model
model.save('yolov8n-seg_yfusion.pt')


# In[ ]:


import glob
import cv2
from matplotlib import pyplot as plt
functional_testing_set = glob.glob('./validation/mprm1-mprm1-entr/*.tif' )
print( functional_testing_set[0 ] )
print(cv2.imread(functional_testing_set[0 ]).shape)

cscheme = [ 'Greys' , 'Greens', 'Reds' ]
sample = cv2.imread(functional_testing_set[0 ])
for channel in range(sample.shape[2] ):
    plt.imshow(sample[:,:,channel] , cmap = cscheme[channel])
    plt.colorbar()
    plt.show()


# In[9]:


import pickle
#my_model = YOLO('yolov8n-seg_yfusion.pt')
my_model = YOLO('yolov8n-seg_yfusion.pt')

#my_model = model
def scaleimg(imgfile):
    #scale each input channel
    im = cv2.imread(imgfile)
    for d in range(im.shape[2]):
        page = im[:,:,d]
        page = (page - page.min()) / (page.max() - page.min()) * 255
        im[:,:,d] = page
    #im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(imgfile +'_scaled.png', im )
    adjust_brightness_contrast(imgfile +'_scaled.png',imgfile +'_scaled.png')
    
    return imgfile +'_scaled.png'

functional_testing_set = glob.glob('./validation/mprm1-mprm1-entr/*.tif' )
predinput = [scaleimg(i) for i in functional_testing_set ]
results = { img:my_model( img , conf=0.0) for img in predinput }
with open( 'functional_test.pkl' , 'wb' ) as resout:
    resout.write( pickle.dumps( results ))


# In[6]:


print(results.keys())


# In[19]:


from matplotlib import patches
import tqdm

def add_rect(h,w,x,y , ax , color = 'r', label= '' ):
    # Create a Rectangle patch
    rect = patches.Rectangle((y,x), h, w, linewidth=3, edgecolor=color, facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    if len(label) > 0:
        ax.text(y - 20, x - 20 , label , )
    return ax

def plot_res(pred, imgfile):
    
    img = pred[0].orig_img
    boxes = pred[0].boxes.xywh.detach().cpu()
    c = pred[0].boxes.cls.detach().cpu()
    print(boxes)
    fig,axes = plt.subplots(nrows=1, figsize=(20,20)  , ncols=3)

    
    cscheme = [ 'Greys' , 'Greens', 'Reds' ]
    sample = cv2.imread(imgfile)
    
    for channel in range(img.shape[2] ):
        axes[channel].imshow(sample[:,:,channel] , cmap = cscheme[channel])
    for row in tqdm.tqdm(range(boxes.shape[0])):
        x,y,w,h = list( boxes[row,: ] )
        cstr = str(c[row])
        label = str( c[row] )
        for ax in list(axes):
            ax = add_rect(h=h,w=w,x=x,y=y , ax=ax , color = 'r', label= cstr )
    plt.show()


def output_pred(imgfile,my_model=my_model):
    results = list(my_model(imgfile, conf=0.01))
    plot_res(results, imgfile)


# In[20]:


output_pred(predinput[0])


# In[21]:


output_pred('datasets/test/images/img_68.png')


# In[ ]:





# In[15]:


#visualize predicitions for the validation set
import random
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tqdm 

my_model = YOLO('runs/segment/train21/weights/best.pt')
results = list(my_model('datasets/test/images/img_57.png', conf=0.0))
result = np.asarray( results[0].conf.detach())
def retdf(model, predfile):
    resdict = { i:{ c:result[row,i] for i,c in enumerate(classes) } for i in result.shape[0] }
    resdf = pd.DataFrame(resdict)
    resdf['file'] = predfile
    return resdf

predfiles = glob.glob('*/')
global_resdf =  pd.concat([ retdf(my_model(imgfile, conf=0.0) ) for imgfile in tqdm.tdm(predfiles) ])

print(global_resdf )


# In[ ]:





# In[16]:


print(results)


# In[17]:


from IPython.display import Image as show_image
show_image(filename="runs/segment/train5/val_batch0_labelsYOLOjpg")

show_image(filename="runs/segment/train5/MaskP_curve.png")

show_image(filename="runs/segment/train5/results.png")



# In[ ]:





# In[ ]:


#augment dataset and dump to disk in pt format

