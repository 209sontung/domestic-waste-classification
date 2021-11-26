# NEU-Bin: Waste classification

Urban waste management has always been a challenging problem due to the increasingly abundant amount of mixed domestic waste without household waste segregation. The remarkable advancement in deep learning helps computer vision systems gain splendid achievements in image classification and image recognition, including image-based waste identification and classification. 
- We separate three significant categories of domestic waste: recyclable waste (plastic, paper, glass-metal), biodegradable waste, and non-recyclable waste.
- Our ResNet50-based proposed model achieves an 87.50% prediction accuracy on the test dataset.

## Prerequisites

### Installing Python and dependencies
* [Python 3 and above](https://www.python.org/downloads/)
* OpenCV 3.4.0 for Python
* Numpy
* Keras
* Tensorflow
* Scikit-learn

## Details
### <a name="dataset"></a> Dataset

The overall dataset includes 3495 images combined from selected data from [`Trashnet`](https://github.com/garythung/trashnet) and `Waste-set` (build by our own). Given the nature of the data and the purpose of research towards solutions contributing to environmental protection and support appropriate supply for recycling plants, the data is classified into three main categories.

-	Recyclable wastes:
    - Plastic: 538
    - Paper/Cardboard: 616
    - Glass & Metal: 849
-	Organic wastes: 487
-	Non-recyclable wastes: 1005

<img src="https://github.com/209sontung/NEU-Bin/blob/main/img/dts.png" alt="alt text" width="600" height="400">

The dataset is available for download [here](https://drive.google.com/file/d/1IUns7XIZjoEeXG0S4szhaSCuceQGujKY/view?usp=sharing)

#### <a name="preparing"></a> Data Augmentation

In our experiments, we use the ImageDataGenerator class1 from Keras to provide several transformations for generating new training data, such as rotation, zooming, translation, randomly flipping images horizontally, and filling new pixels with their nearest surround pixels.

<!-- :warning: You may use *additional_dataset.zip* as another version of dataset. But if you use both of them on training phase, it will increase intra-class variance thus will leads to decrease of accuracy. Maybe you can try to use it for just testing true-generalizability on totally different dataset.(In terms of real world problem, trashes have high intra-class variance so it's very important!) -->

## Proposed model
<img src="https://github.com/209sontung/NEU-Bin/blob/main/img/nbin.png" alt="alt text" width="600" height="300">

The process of building NEU-Bin consists of two stages:
* Phase 1: Since the layers of the pre-trained model has been trained on the ImageNet dataset, we freeze the classes of the ResNet50 model and only update the weights of added layers. When the loss function becomes more stable, and the network reaches a higher level of accuracy with the added layers, we continue to the next phase.
* Phase 2: At this stage, we unfreeze the last few layers of the pre-trained model and continue training with these layers along with the newly added adjustment layers. 

## Experimental result

|         **Model**       |   **Accuracy (%)**   |     **Parameters (M)**     |
|-------------------------|:--------------------:|:--------------------------:|
|       ResNet50          |        87.50         |            23.7            |
|       DenseNet121       |        86.50         |            7.10            |
|       MobileNetV2       |        83.40         |            2.34            |
|       VGG16             |        82.30         |            14.7            |
|       InceptionV3       |        82.50         |            21.9            |

## Usage
#### Download and unzip
```
$ git clone https://github.com/209sontung/NEU-Bin.git
```
You can download the trained model [here](https://drive.google.com/file/d/1wamwLZsclQYYsx5dLThTqZG5sJSLR7oS/view?usp=sharing)

#### Change the model
```
model = load_model('model_5class_resnet_87%.h5')
```
#### Other minor changes

* To change input camera (0 means built-in camera):
```
cap = cv2.VideoCapture(0)
```

* To change threshold:
```
threshold = 0.85 
```
Lower the threshold, lower the confidence of the model

## Run the code and start detection
Run the code and the detection will start. Hit `Q` to exit.

<img src="https://github.com/209sontung/NEU-Bin/blob/main/img/examples.png" alt="alt text" width="600" height="800">

## Contact
- Supervisor: [Tuan Nguyen](https://www.facebook.com/nttuan8)
- Team Members: [Tung Nguyen](https://www.facebook.com/gnutn0s), [Duc Ha](https://www.facebook.com/ha5minh2duc), [Ha Phuong Dinh](https://www.facebook.com/profile.php?id=100008189945262)



