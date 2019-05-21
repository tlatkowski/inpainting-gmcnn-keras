![](https://img.shields.io/badge/Python-3.6-blue.svg) ![](https://img.shields.io/badge/Keras-2.2.4-blue.svg) ![](https://img.shields.io/badge/TensorFlow-1.12.0-blue.svg) ![](https://img.shields.io/badge/License-MIT-blue.svg)

# Generative Multi-column Convolutional Neural Networks inpainting model in Keras
**Keras** implementation of **GMCNN** (Generative Multi-column Convolutional Neural Networks) inpainting model originally proposed at NIPS 2018:
[Image Inpainting via Generative Multi-column Convolutional Neural Networks](https://arxiv.org/pdf/1810.08771.pdf)


## Model architecture
![GMCNN model](./pics/models/gmcnn_model.png)

![](https://img.shields.io/badge/-convolution-yellow.svg)
![](https://img.shields.io/badge/-dilated_convolution-red.svg)
![](https://img.shields.io/badge/-up_scaling-blue.svg)
![](https://img.shields.io/badge/-concatention-lightgrey.svg)
## Installation
* Code from this repository was tested on **Python 3.6** and **Ubuntu 14.04**
* All required dependencies are stored in **requirements.txt**, **requirements-cpu.txt** and **requirements-gpu.txt** files.

Code download:
```bash
git clone https://github.com/tlatkowski/inpainting-gmcnn-keras.git
cd inpainting-gmcnn-keras
```

To install requirements, create Python virtual environment and install dependencies from files:
```bash
virtualenv -p /usr/bin/python3.6 .venv
source .venv/bin/activate
pip install -r requirements/requirements.txt
```
In case of using GPU support:
```bash
pip install -r requirements/requirements-gpu.txt
```
Otherwise (CPU usage):
```bash
pip install -r requirements/requirements-cpu.txt
```


## Datasets

### Image dataset
Model was trained with usage of high-resolution images from Places365-Standard dataset.
It can be found [here](http://places2.csail.mit.edu/download.html)


### Mask dataset
The mask dataset used for model training comes from NVIDIA's paper: [Image Inpainting for Irregular Holes Using Partial Convolutions](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Guilin_Liu_Image_Inpainting_for_ECCV_2018_paper.pdf)

NVIDIA's mask dataset is available [here](http://masc.cs.gmu.edu/wiki/partialconv)

**Please note that the model training was performed on testing irregular mask dataset containing  12,000 masks.**
 
 
**./samples** folder contains exemplary structure of dataset catalogs:
```bash
samples
 |-masks
    |-nvidia_masks
 |-images
    |-places365
```
**nvidia_masks** catalog contains 5 sample masks from NVIDIA's test set.

**places365** catalog contains 5 sample images form Places365 validation set.
## Model training
The main configuration file is placed in **./config/main_config.ini**. It contains training and model parameters. You can tweak those parameters before model running.

The default configuration looks as follows:
```ini
[TRAINING]
WGAN_TRAINING_RATIO = 5
NUM_EPOCHS = 5
BATCH_SIZE = 4
IMG_HEIGHT = 256
IMG_WIDTH = 256
NUM_CHANNELS = 3
LEARNING_RATE = 0.0001
SAVE_MODEL_STEPS_PERIOD = 1000

[MODEL]
ADD_MASK_AS_GENERATOR_INPUT = False
GRADIENT_PENALTY_LOSS_WEIGHT = 10
ID_MRF_LOSS_WEIGHT = 0.05
ADVERSARIAL_LOSS_WEIGHT = 0.001
NN_STRETCH_SIGMA = 0.5
VGG_16_LAYERS = 3,6,10
ID_MRF_STYLE_WEIGHT = 1.0
ID_MRF_CONTENT_WEIGHT = 1.0
NUM_GAUSSIAN_STEPS = 3
GAUSSIAN_KERNEL_SIZE = 32
GAUSSIAN_KERNEL_STD = 40.0
```

After the dependencies installation you can perform training dry-run using image and mask samples provided in **samples** directory. To do so, execute the following command:

**NOTE: Set BATCH_SIZE to 1 before executing the below command.**
```bash
python runner.py --train_path ./samples/images --mask_path ./samples/masks --experiment_name "dry-run-test"
```
If everything goes correct you should be able to see the progress bar logging the basic training metrics.

In order to run GMCNN model training on your training data you have to provide paths to your datasets:
```bash
python runner.py --train_path /path/to/training/images --mask_path /path/to/mask/images --experiment_name "experiment_name"
```

### Warm-up generator training
According to the best practices of the usage of GAN frameworks, first we should train the generator model for a while. In order to train the generator only in the first line run the following command (additional flag **warm_up_generator** is set):
```bash
python runner.py --train_path /path/to/training/images --mask_path /path/to/mask/images -warm_up_generator
```
In this mode the generator will be trained with only confidence-driven reconstruction loss.

Below picture presents GMCNN outcome after 5 epochs training in warm-up generator mode
![](https://github.com/tlatkowski/inpainting-gmcnn/blob/master/pics/outputs/warm_up_generator_5_epochs.jpg)

### WGAN-GP training
In order to continue training with full WGAN-GP framework (GMCNN generator, local and global discriminators), execute:
```bash
python runner.py --train_path /path/to/training/images --mask_path /path/to/mask/images --experiment_name "experiment_name" -from_weights
```

Running training with additional **from_weights** flag will force pipeline to load the latest models checkpoints from **./outputs/weights/** directory. 

### GMCNN model training in Google Colab notebook
If you don't have an access to workstation with GPU, you can use the below exemplary Google Colab notebook for training your GMCNN model on Places365 validation data and NVIDIA's testing mask with usage of K80 GPU available within Google Colab backend: [GMCNN in Google Colab](https://github.com/tlatkowski/inpainting-gmcnn-keras/blob/master/colab/Image_Inpainting_with_GMCNN_model.ipynb)



### Pipeline outcomes

During the training procedure the pipeline logs additional results to the **outputs** directory:
* **outputs/experiment_name/logs** contains TensorBoard logs
* **outputs/experiment_name/predicted_pics/warm_up_generator** contains the model predictions for the specific steps in the warm up generator training mode
* **outputs/experiment_name/predicted_pics/wgan** contains the model predictions for the specific steps in the WGAN-GP training mode
* **outputs/experiment_name/weights** contains the generator and critics models weights
* **outputs/experiment_name/summaries** contains the generator and critics models summaries

You can track the metrics during the training with usage of TensorBoard:
```bash
tensorboard --logdir=./outputs/experiment_name/logs
```

## Implementation differences from original paper

1. This model is trained using NVIDIA's irregular mask test set whereas the original model is trained using randomly generated rectangle masks. 
2. The current version of pipeline uses the higher-order features extracted from VGG16 model whereas the original model utilizes VGG19.

## Visualization of Gaussian blurring masks 

Below you can find the visualization of applying Gaussian blur to the training masks for the different number of convolution steps (number of iteration steps over the input raw mask). 

#### Large mask
Original | 1 step | 2 steps | 3 steps | 4 steps | 5 steps | 10 steps
------- |  ------- | ------- | ------- | ------- | ------- | ------- 
![](https://github.com/tlatkowski/inpainting-gmcnn/blob/master/pics/masks/large_mask_original.png) | ![](https://github.com/tlatkowski/inpainting-gmcnn/blob/master/pics/masks/large_blurred_mask_1_step.png) | ![](https://github.com/tlatkowski/inpainting-gmcnn/blob/master/pics/masks/large_blurred_mask_2_step.png) | ![](https://github.com/tlatkowski/inpainting-gmcnn/blob/master/pics/masks/large_blurred_mask_3_step.png) | ![](https://github.com/tlatkowski/inpainting-gmcnn/blob/master/pics/masks/large_blurred_mask_4_step.png) | ![](https://github.com/tlatkowski/inpainting-gmcnn/blob/master/pics/masks/large_blurred_mask_5_step.png) | ![](https://github.com/tlatkowski/inpainting-gmcnn/blob/master/pics/masks/large_blurred_mask_10_step.png)


#### Small mask
Original | 1 step | 2 steps | 3 steps | 4 steps | 5 steps | 10 steps
------- | ------- | ------- | ------- | ------- | ------- | -------  
![](https://github.com/tlatkowski/inpainting-gmcnn/blob/master/pics/masks/small_mask_original.png) | ![](https://github.com/tlatkowski/inpainting-gmcnn/blob/master/pics/masks/small_blurred_mask_1_step.png) | ![](https://github.com/tlatkowski/inpainting-gmcnn/blob/master/pics/masks/small_blurred_mask_2_step.png) | ![](https://github.com/tlatkowski/inpainting-gmcnn/blob/master/pics/masks/small_blurred_mask_3_step.png) | ![](https://github.com/tlatkowski/inpainting-gmcnn/blob/master/pics/masks/small_blurred_mask_4_step.png) | ![](https://github.com/tlatkowski/inpainting-gmcnn/blob/master/pics/masks/small_blurred_mask_5_step.png) | ![](https://github.com/tlatkowski/inpainting-gmcnn/blob/master/pics/masks/small_blurred_mask_10_step.png)

#### Rectangle mask
Original | 1 step | 2 steps | 3 steps | 4 steps | 5 steps | 10 steps
------- | ------- | ------- | ------- | ------- | ------- | ------- 
![](https://github.com/tlatkowski/inpainting-gmcnn/blob/master/pics/masks/rectangle_mask_original.png) | ![](https://github.com/tlatkowski/inpainting-gmcnn/blob/master/pics/masks/rectangle_blurred_mask_1_step.png) | ![](https://github.com/tlatkowski/inpainting-gmcnn/blob/master/pics/masks/rectangle_blurred_mask_2_step.png) | ![](https://github.com/tlatkowski/inpainting-gmcnn/blob/master/pics/masks/rectangle_blurred_mask_3_step.png) | ![](https://github.com/tlatkowski/inpainting-gmcnn/blob/master/pics/masks/rectangle_blurred_mask_4_step.png) | ![](https://github.com/tlatkowski/inpainting-gmcnn/blob/master/pics/masks/rectangle_blurred_mask_5_step.png) | ![](https://github.com/tlatkowski/inpainting-gmcnn/blob/master/pics/masks/rectangle_blurred_mask_10_step.png)


## Visualization of training losses

After activating TensorBoard you can monitor the following training metrics:
1. For the generator: confidence reconstruction loss, global wasserstein loss, local wasserstein loss, id mrf loss and total loss
2. For the local and global discriminators: fake loss, real loss, gradient penalty loss and total loss

![](https://github.com/tlatkowski/inpainting-gmcnn/blob/master/pics/tb_log.png)

## Code References

1. ID-MRF loss function was implemented with usage of original Tensorflow implementation: [GMCNN in Tensorflow](https://github.com/shepnerd/inpainting_gmcnn)
2. Improved Wasserstain GAN was implemented based on: [Wasserstein GAN with gradient penalty in Keras](https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py)
3. Model architecture diagram was done with usage of PlotNeuralNet: [PlotNeuralNet on GitHub](https://github.com/HarisIqbal88/PlotNeuralNet)
