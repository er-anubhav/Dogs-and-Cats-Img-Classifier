
# Dogs vs. Cats Image Classifier

This guide details the workflow implemented in a Colab notebook to create a Convolutional Neural Network (CNN) for classifying images of dogs and cats. The model uses TensorFlow and Keras for defining, training, and validating a deep learning model.

## Table of Contents
1. [Setup and Dataset Download](#setup)
2. [Data Preprocessing](#preprocessing)
3. [Model Building](#model-building)
4. [Model Compilation](#model-compilation)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [Conclusion](#conclusion)

<a name="setup"></a>
## 1. Setup and Dataset Download

The first step is to set up the environment and download the dataset from Kaggle. Ensure that Kaggle API keys are configured in Colab for this process.

```python
# Configuring Kaggle API for downloading datasets
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Downloading the dataset
!kaggle datasets download salader/dogs-vs-cats

# Unzipping the dataset
import zipfile
zip_ref = zipfile.ZipFile('/content/dogs-vs-cats.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()
```

This code configures the environment to use Kaggle, downloads the "Dogs vs. Cats" dataset, and extracts it for use.

<a name="preprocessing"></a>
## 2. Data Preprocessing

### Image Loading and Batch Creation
Using TensorFlow’s `image_dataset_from_directory`, images are loaded from directories and organized into batches for training and validation.

```python
train_ds = keras.utils.image_dataset_from_directory(
    directory = '/content/train',
    labels = 'inferred',
    label_mode = 'int',
    batch_size = 32,
    image_size = (256, 256)
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory = '/content/test',
    labels = 'inferred',
    label_mode = 'int',
    batch_size = 32,
    image_size = (256, 256)
)
```

### Normalization
To enhance model training, images are normalized by scaling pixel values to the range `[0,1]`.

```python
def process(image, label):
    image = tf.cast(image/255., tf.float32)
    return image, label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)
```

Normalization helps in stabilizing and accelerating the training process by ensuring consistent input values.

<a name="model-building"></a>
## 3. Model Building

A Convolutional Neural Network (CNN) model is defined for binary classification. The architecture includes layers for feature extraction (convolution and pooling) and dense layers for classification.

```python
model = Sequential()

# Convolution and Pooling Layers
model.add(Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(256, 256, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

model.add(Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

# Flattening and Fully Connected Layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
```

This model uses three convolutional layers with ReLU activation, followed by batch normalization and max-pooling, which helps in feature extraction. Flattening is applied before passing the features to the dense layers for classification.

<a name="model-compilation"></a>
## 4. Model Compilation

The model is compiled with the `adam` optimizer, binary cross-entropy loss (suitable for binary classification), and accuracy as the evaluation metric.

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

The Adam optimizer dynamically adjusts learning rates during training, while binary cross-entropy calculates the loss for binary classification.

<a name="model-training"></a>
## 5. Model Training

The model is trained using the training dataset for a specified number of epochs. During training, model performance is tracked on the validation dataset.

```python
history = model.fit(train_ds, validation_data=validation_ds, epochs=10)
```

This step provides feedback on model accuracy and loss after each epoch, allowing you to monitor convergence.

<a name="model-evaluation"></a>
## 6. Model Evaluation

After training, the model can be evaluated on unseen test data or validation data to assess its accuracy and generalization capability.

```python
model.evaluate(validation_ds)
```

Evaluation provides metrics such as accuracy, enabling a performance check on new data that the model has not encountered during training.

<a name="conclusion"></a>
## Conclusion

This notebook provides an end-to-end approach to building a simple CNN-based binary image classifier for distinguishing between dog and cat images. You can improve this model by experimenting with more complex architectures, data augmentation, or pre-trained models like VGG16 or ResNet to boost accuracy.

## Note
This code is implemented in a Google Colab environment. To execute, ensure that your Kaggle API key (`kaggle.json`) is uploaded to the Colab environment, and GPU acceleration is enabled for faster training.
