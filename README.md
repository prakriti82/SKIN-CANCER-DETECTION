# Skin Cancer Detection System

This project provides a Python-based approach for detecting skin cancer from both images and live video feeds. Utilizing deep learning techniques, the system classifies skin lesions into cancerous and non-cancerous categories.

## Model Architecture

The implemented Convolutional Neural Network (CNN) consists of multiple layers designed for feature extraction and classification:

```python
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

## Dataset Information

The dataset used for training is available on Kaggle: [Skin Cancer Binary Classification Dataset](https://www.kaggle.com/datasets/kylegraupe/skin-cancer-binary-classification-dataset). This dataset includes labeled skin lesion images for binary classification.

## Prerequisites

To set up and run this project, install the necessary dependencies:

```shell
pip install tensorflow keras numpy opencv-python
```

Ensure you have Python 3.x installed before running the above command.

## How to Use

1. Clone this repository:

```shell
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

2. Download the dataset from the given Kaggle link and store it in the relevant directory.

3. Train the model using the provided script.

4. To perform skin cancer detection on an image:

```shell
python predict_image.py --image path/to/image.jpg
```

5. To analyze a video file:

```shell
python predict_video.py --video path/to/video.mp4
```

Replace `path/to/image.jpg` and `path/to/video.mp4` with actual file locations.

## Performance & Results

The trained CNN model provides effective classification of skin lesions with high accuracy. Users can fine-tune the model by adjusting hyperparameters or modifying the architecture for improved performance.

## Credits

- The dataset used in this project is sourced from Kaggle: [Skin Cancer Binary Classification Dataset](https://www.kaggle.com/datasets/kylegraupe/skin-cancer-binary-classification-dataset).



