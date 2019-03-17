![](https://img.shields.io/badge/Python-3.6-blue.svg) ![](https://img.shields.io/badge/Keras-2.2.4-blue.svg) ![](https://img.shields.io/badge/TensorFlow-1.12.0-blue.svg) ![](https://img.shields.io/badge/License-MIT-blue.svg)

# Generative Multi-column Convolutional Neural Networks inpainting model in Keras
**Keras** implementation of **GMCNN** (Generative Multi-column Convolutional Neural Networks) inpainting model originally proposed at NIPS 2018:
[Image Inpainting via Generative Multi-column Convolutional Neural Networks](https://arxiv.org/abs/1810.0877)


## Model architecture
![GMCNN model](./pics/models/gmcnn_model.png)

## Dependencies
* Code on this repository was tested on **Python 3.6** and **Ubuntu 14.04**
* All required dependencies are stored in **requirements.txt**, **requirements-cpu.txt** and **requirements-gpu.txt** files.

Code download:
```bash
git clone https://github.com/tlatkowski/inpainting-gmcnn.git
cd inpainting-gmcnn
```

To install requirements, create Python virtual environment and install requirements from files:
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
The main configuration file is placed at **./config/main_config.ini**. It contains training and model parameters. You can tweak those parameters before model running.

The default configuration looks as follows:
```ini
[TRAINING]
WGAN_TRAINING_RATIO = 5
NUM_EPOCHS = 500
BATCH_SIZE = 1
IMG_HEIGHT = 128
IMG_WIDTH = 128
NUM_CHANNELS = 3
LEARNING_RATE = 0.0001
SAVE_MODEL_EPOCH_PERIOD = 50

[MODEL]
GRADIENT_PENALTY_LOSS_WEIGHT = 10
ID_MRF_LOSS_WEIGHT = 0.05
ADVERSARIAL_LOSS_WEIGHT = 0.001
NUM_GAUSSIAN_STEPS = 3
```

After dependencies installation you can perform dry-run using image and mask samples provided in **samples** directory. To do so, execute the following command:
```bash
python runner.py --train_path ./samples/images --mask_path ./samples/masks
```
If everything went correct you should be able to see progress bar logging basic training metrics.

To run GMCNN model training on your training data you have to provide paths to your datasets:
```bash
python runner.py --train_path /path/to/training/images --mask_path /path/to/mask/images
```

To continue training you should add **-from_weights** additional flag to training runner:
```bash
python runner.py --train_path /path/to/training/images --mask_path /path/to/mask/images -from_weights
```

### Warm-up generator training
According to the best practices of the usage of GAN frameworks, first we should train the generator model for a while. In order to train the generator only in the first line run the following command (additional flag **warm_up_generator** is set):
```bash
python runner.py --train_path /path/to/training/images --mask_path /path/to/mask/images -warm_up_generator
```
In this mode the generator will be trained with only confidence-driven reconstruction loss.

### WGAN-GP training
In order to continue training with full WGAN-GP framework (GMCNN generator, local and global discriminator), execute:
```bash
python runner.py --train_path /path/to/training/images --mask_path /path/to/mask/images -from_weights
```

During the training procedure the pipeline logs additional results to the **outputs** directory:
* **outputs/logs** contains TensorBoard logs
* **outputs/predicted_pics/warm_up_generator** contains the model outputs for specific epochs in the warm up generator training mode
* **outputs/predicted_pics/wgan** contains the model outputs for specific epochs in the WGAN training mode
* **outputs/weights** contains the generator and critics model weights
* **outputs/summaries** contains the generator and critics model summaries

You can track the metrics during training with usage of TensorBoard:
```bash
tensorboard --logdir=./outputs/logs
```

## Implementation differences from original paper

1. This model is trained using NVIDIA's irregular mask test set whereas the original model is trained using randomly generated rectangle masks. 
2. Current version of pipeline uses the higher-order features extracted from VGG16 model where as the original model utilizes VGG19.

## Visualization of Gaussian blurring masks 

Below you can find the visualisation of applying Gaussian blur to the training masks for the different number of convolution steps (number of iteration steps over the input raw mask). 

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


## Code References

1. ID-MRF loss function was implemented with usage of original Tensorflow implementation: [GMCNN in Tensorflow](https://github.com/shepnerd/inpainting_gmcnn)
2. Improved Wasserstain GAN was implemented based on: [Wasserstein GAN with gradient penalty in Keras](https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py)
3. Model architecture diagram was done with usage of PlotNeuralNet: [PlotNeuralNet on GitHub](https://github.com/HarisIqbal88/PlotNeuralNet)
