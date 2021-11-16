# NEU-Bin: Waste classification

Urban waste management has always been a challenging problem due to the increasingly abundant amount of mixed domestic waste without household waste segregation. The remarkable advancement in deep learning helps computer vision systems gain splendid achievements in image classification and image recognition, including image-based waste identification and classification. 
- We separate three significant categories of domestic waste: recyclable waste (plastic, paper, glass-metal), biodegradable waste, and non-recyclable waste.
- Our ResNet50-based proposed model achieves an 87.50% prediction accuracy on the test dataset.

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

The dataset is available for download [here](https://drive.google.com/drive/folders/1KRGZJIS_D1aCTdalDG5ppjrW9HtQtCsC?usp=sharing)

#### <a name="preparing"></a> Data Augmentation

In our experiments, we use the ImageDataGenerator class1 from Keras to provide several transformations for generating new training data, such as rotation, zooming, translation, randomly flipping images horizontally, and filling new pixels with their nearest surround pixels.
