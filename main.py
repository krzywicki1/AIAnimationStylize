import time
start_time = time.time()
from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub
import glob
import PIL
import imageio
from pathlib import Path

print("--- %s seconds ---" % round(time.time() - start_time,2))

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU'))) #Check if Tensorflow is working on GPU


#Load image
def load_image(image_path, image_size=(1280, 720)):
    img = tf.io.decode_image(
      tf.io.read_file(image_path),
      channels=3, dtype=tf.float32)[tf.newaxis, ...]
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img

#Create window with images /// Not using rn
def visualize(images):
    noi = len(images)
    image_sizes = [image.shape[1] for image in images]
    w = (image_sizes[0] * 6) // 320
    plt.figure(figsize=(w  * noi, w))
    grid_look = gridspec.GridSpec(1, noi, width_ratios=image_sizes)
    
    for i in range(noi):
        plt.subplot(grid_look[i])
        plt.imshow(images[i][0], aspect='equal')
        plt.axis('off')
        plt.savefig("final.jpg")
    
    plt.show()

#Export image
def export_image(tf_img):
    tf_img = tf_img*255
    tf_img = np.array(tf_img, dtype=np.uint8)
    if np.ndim(tf_img)>3:
        assert tf_img.shape[0] == 1
        img = tf_img[0]
    return PIL.Image.fromarray(img)

#Load list of images
original_images = [load_image(file) for file in glob.glob("original/*.jpg")]
#Load style for ai
style_image = load_image("style.jpg")
#Learning ai with style
#Ksize - bigger - smoother/blursed // 
#Strides - smaller - better quality // if you want longer strides give one vector more
#Must be int
style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='VALID') # 3,3 1,1 - best result
#model for stylizing use magenta-arbitrary-stylization
stylize_model = tf_hub.load('tf_model')

print('\n\n')
#Loop for creating images
i = 0
for x in original_images:
    print('Processing ' + str(i+1)+'/'+str(len(original_images))+ ' image')
    results = stylize_model(tf.constant(x), tf.constant(style_image))
    stylized_photo = results[0]
    export_image(stylized_photo).save("result/%i.png" %i)
    i=i+1



print('\n\n')

print('DONE in %s seconds! Check your ./result folder' % round(time.time() - start_time,2))