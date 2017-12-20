# **SPP_net**

## **Brief** 

Tradition CNNs require a fixed input size, that is the reason why it need crop/warp images fisrt.

This requirement is “artificial” and may reduce the recognition accuracy for the images or sub-images of an arbitrary size/scale. 
### **Traditional CNNs**
image -> crop/warp -> conv layers -> fc layers -> output
### **SPP-net**
image -> conv layers -> spatial pyramid pooling -> fc layers -> output



## **Why do CNNs require a fixed input size?**

The convolutional layers operate in a sliding-window manner and output feature maps which represent the spatial arrangement of the activation.

In fact convolutional layers do not require a fixed image size and it can generate feture maps of any sizes.

On the other hand,the fixed size constraint comes only from the fully-connected layers.

## **The Spatial Pyramid Pooling (SPP) Layer**

*Spatial Pyramid Pooling* is another pooling method conpared the normal pooling layer (eg. 2x2 maximum pooling layer).

The output feature map size (image size) of normal pooling layer depends on the input feature map size. Normally it appoint the maximum value of 4 Neighboring pixels at the output pixel value. As a consequence, it will reduce the feature map size with little information loss.

For *Spatial Pyramid Pooling*, it consider the output feature number. It also will reduce the feature map size, but the output size depends on the SPP layer. As a figure shown below, there are three spp filters(4x4, 2x2, 1x1).

At the first layer, it divide the whole input feature map into 16 part. Provided the length and width of feature map is W and H. The length and width of the bins are (L/4, W/4).

At the second layer, it divide the whole input feature map into 4 part. The length and width of the bins are (L/2, W/2).

At the third layer, The length and width of the bins are (L, W).

If we use maximum Spatial Pyramid Pooling, the filter will get the maximum value of each bins' pixels to represent each bins. So the output feature number of first layer is 16, the output feature number of second layer is 4, and the third is 1. 

No matter what size of the input feature map, it will output the fixed number of feature which is very important for the fully connected layer.

![SPP](/home/binzhang/Pictures/SPP.png)


## **Experiment part**

In practice the GPU implementations (such as *cuda-convnet* and  *Caffe*) are preferably run on fixed input images.

## **Multi-size training** 

![tfcnn](/home/binzhang/Pictures/tfcnn.png)

During traing they implement the varying-input-size SPP-net by two fixed-size networks that share parameters.

(Share parameters means that the weight of the conv layer and fc layer is same, just the define of the input image size of conv layer is different)

To reduce the overhead to switch from one (eg. 224x224) network to other (eg. 180x180) , they train each full epoch on one network, and then switch to the other one (keeping all weights) for the next full epoch.

Note that the above multi-size solutions are for training only. At the testing stage, it is straightforward to apply SPP-net on images of any sizes.

## **Result of image classification**

### **Multi-level Pooling Improves Accuracy**

In these networks, the convolutional layers
have the same structures as the corresponding baseline models, whereas the pooling layer after the final convolutional layer is replaced with the SPP layer. 

### **Multi-size Training Improves Accuracy**

Their results using multi-size training. The training sizes are 224 and 180, while the testing size is still 224. 

### **Full-image Representations Improve Accuracy**

## **SPP-NET FOR OBJECT DETECTION**

For Object Detection, they also need *Selective Search* method to extract candidate windows from the input image.

### **Training Part** 

The CNN part and the SVM part is trained separately.

CNN part is trained to extract the feature of the input image. For those image classification and object detection, the weight of the CNN layer is same. That is because the CNN is to get the feature which is similar to it's convolutional core. The feature of object is same between image classification and object detection.

They use the 'fast' mode of selective search method to generate about 2,000 candidate windows per image. Ground-truth window is used to generate possitive samples for SVM training. The negative samples are those overlapping a positive window by at most 30% (measured by the intersection-over-union (IoU) ratio). Any negative sample is removed if it overlaps another negative sample by more than 70%

After that they use spatial pyramid pooling on those window and get the feature to train SVM.

### **Testing Part**
Input image -> CNNs -> generate windows -> SPP layers -> SVM

![tfcnn](/home/binzhang/Pictures/SPP_Ob_Det.png)
