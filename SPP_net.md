# **SPP_net**
Tradition CNNs require a fixed input size, that is the reason why it need crop/warp images fisrt.

This requirement is “artificial” and may reduce the recognition accuracy for the images or sub-images of an arbitrary size/scale. 
### Traditional CNNS
image -> crop/warp -> conv layers -> fc layers -> output
### SPP-net
image -> conv layers -> spatial pyramid pooling -> fc layers -> output



## Why do CNNs require a fixed input size?

The convolutional layers operate in a sliding-window manner and output feature maps which represent the spatial arrangement of the activation.

In fact convolutional layers do not require a fixed image size and it can generate feture maps of any sizes.

On the other hand,the fixed size constraint comes only from the fully-connected layers.

## The Spatial Pyramid Pooling Layer
(reserve)

## Experiment part 

In practice the GPU implementations (such as *cuda-convnet* and  *Caffe*) are preferably run on fixed input images.

## Multi-size training 

During traing they implement the varying-input-size SPP-net by two fixed-size networks that share parameters.

(Share parameters means that the weight of the conv layer and fc layer is same, just the define of the input image size of conv layer is different)

To reduce the overhead to switch from one (eg. 224x224) network to other (eg. 180x180) , they train each full epoch on one network, and then switch to the other one (keeping all weights) for the next full epoch.

Note that the above multi-size solutions are for training only. At the testing stage, it is straightforward to apply SPP-net on images of any sizes.

## Result of image classification

### Multi-level Pooling Improves Accuracy

In these networks, the convolutional layers
have the same structures as the corresponding baseline models, whereas the pooling layer after the final convolutional layer is replaced with the SPP layer. 

### Multi-size Training Improves Accuracy

Their results using multi-size training. The training sizes are 224 and 180, while the testing size is still 224. 

 ### Full-image Representations Improve Accuracy






 intersection-over-union (IoU) 

 Q1: How Spatial Pyramid Pooling Layer works?

 Q2: How the ground-truth window used?