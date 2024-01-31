import os
import numpy as np
import time
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Define the model input and output directory
DATA_DIR = "./data/CIFAR10/"
SAVE_DIR = "./results/CIFAR10"

# Make the save dirs if they does not exist
os.makedirs(os.path.join(SAVE_DIR), exist_ok=True)
cp_dir = os.path.join(SAVE_DIR, "model_checkpoints")
os.makedirs(cp_dir, exist_ok=True)
arc_dir = os.path.join(SAVE_DIR, "model_architectures")
os.makedirs(arc_dir, exist_ok=True)
dataset_dir = os.path.join(DATA_DIR, "dataset")

# CIFAR10 comes as part of TF Keras
CIFAR10 = tf.keras.datasets.cifar10

# Load the images
# Normally the entire dataset wouldn't be loaded into memory, but this is small
#   enough that it's ok to do so
non_test, test = CIFAR10.load_data()
non_test_images, non_test_labels = non_test
test_images, test_labels = test

# Preprocess images (zero-mean and normalized). Use non-test data for test
#   data normalization since for most applications the test data is not
#   available all at once to normalize
mean_image = np.mean(non_test_images, axis=0)
non_test_images = non_test_images.astype(float)
test_images  = test_images.astype(float)
non_test_images -= mean_image
test_images -= mean_image
non_test_images /= 255.
test_images /= 255.

# Pulled from the Keras (https://keras.io/api/datasets/cifar10/)
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 
           'horse', 'ship', 'truck']

# Get the input shapes for the networks
input_shape = non_test_images[0].shape

# Define a few models for testing. This will be an exercise in model complexity
#   and generally present the trade-off between inference time and performance.
#   All models were trained with an initial learning rate of 1e-3 that was
#   reduced to 1e-6 as the validation performance plateaus. The training
#   stopped after a long plateau, and the weights are reverted to the model
#   with the highest categorical accuracy on the validation data. During
#   training, for regularization all models had 20% dropout between the last
#   fully connected layer and the layer prior to that.

# Create a simple convolutional network with regularization (dropout)
cnn_simple = tf.keras.Sequential([
    tf.keras.layers.ZeroPadding2D(padding=(2, 2), input_shape=input_shape),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Softmax()
])

# Create a deep convolutional network
cnn_deep = tf.keras.Sequential([
    tf.keras.layers.ZeroPadding2D(padding=(2, 2), input_shape=input_shape),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.ZeroPadding2D(padding=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Softmax()
])

# Create a deep convolutional network
cnn_deeper = tf.keras.Sequential([
    tf.keras.layers.ZeroPadding2D(padding=(2, 2), input_shape=input_shape),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.ZeroPadding2D(padding=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.ZeroPadding2D(padding=(2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Softmax()
])

# Create ResNet-based models. For this, we will use ResNet18, and as well as
#   a modified version of it that uses the first half of it to construct a
#   ResNet10. The ResNet18 uses pretrained weights from training on ImageNet.
#   The final fully connected layer is replaced with a new one with only 10
#   outputs for our 10 classes of CIFAR10.
# Define ResNet18 for CIFAR10
resnet18_imagenet = tf.keras.models.load_model(
    os.path.join(DATA_DIR, "ResNet18.h5"),
    compile=False
)
resnet18_imagenet = tf.keras.models.Model(
    resnet18_imagenet.input,
    resnet18_imagenet.layers[-3].output,
    name="ResNet18"
)
resnet18 = tf.keras.Sequential(resnet18_imagenet)
resnet18.add(tf.keras.layers.Dropout(0.2))    
resnet18.add(
    tf.keras.layers.Dense(
        units=10,
        name="fc_10",
        activation=None,
        input_shape=resnet18.layers[-1].output[1]
    )
)
resnet18.add(tf.keras.layers.Softmax(name="softmax"))

# Define ResNet10 for CIFAR10
resnet18_imagenet = tf.keras.models.load_model(
    os.path.join(DATA_DIR, "ResNet18.h5"),
    compile=False
)
resnet10_imagenet = tf.keras.models.Model(
    resnet18_imagenet.input,
    resnet18_imagenet.layers[47].output,
    name="ResNet9"
)
resnet10 = tf.keras.Sequential(resnet10_imagenet)
resnet10.add(tf.keras.layers.GlobalAveragePooling2D())
resnet10.add(tf.keras.layers.Dropout(0.2))    
resnet10.add(
    tf.keras.layers.Dense(
        units=10,
        name="fc_10",
        activation=None,
        input_shape=resnet10.layers[-1].output[1]
    )
)
resnet10.add(tf.keras.layers.Softmax(name="softmax"))


# Create a data augmentation object
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=0,
    width_shift_range=0.05,
    height_shift_range=0.05,
    brightness_range=[0.95, 1.05],
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    vertical_flip=False,
    validation_split=0.1
)
train_generator = datagen.flow(
    non_test_images, non_test_labels, batch_size=64, subset='training'
)
val_generator = datagen.flow(
    non_test_images, non_test_labels, batch_size=128, subset='validation'
)

# Create a callback to reduce learning rate on plateau
# If the model plateaus for 10 epochs, the learning rate is reduced
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_sparse_categorical_accuracy",
    factor=0.1,
    patience=5,
    verbose=0,
    mode="auto",
    min_delta=0.01,
    cooldown=0,
    min_lr=1e-6,
)

# Create a callback to stop the training on a long plateau
# If the model does not improve for 20 iterations, stop the training
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_sparse_categorical_accuracy",
    min_delta=0.01,
    patience=10,
    verbose=0,
    mode="auto",
    baseline=None,
    start_from_epoch=0,
)

# Line up the models and train each of them
models = [ cnn_simple,   cnn_deep,   cnn_deeper,   resnet10,   resnet18]
names =  ["cnn_simple", "cnn_deep", "cnn_deeper", "resnet10", "resnet18"] 

# Train each model
histories = []
for model, name in zip(models, names):
    # Plot the model architecture
    tf.keras.utils.plot_model(
        model,
        os.path.join(arc_dir, name + ".png"),
        show_shapes=True,
        expand_nested=True,
        dpi=250
    )

    # Optimizer. Initialize the learning rate to 1e-3. beta_1 and beta_2 are
    #   the exponential decay rates of the optimizer (effectively momentum and
    #   RMSProp terms)
    opt = tf.keras.optimizers.Adam(
        learning_rate=1e-3,
        beta_1=0.9,
        beta_2=0.999,
    )

    # Complite the model with the optimizer
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['sparse_categorical_accuracy'],
        optimizer=opt
    )

    # Set a checkpoint to save the data halfway through
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            cp_dir,
            name + ".{epoch:02d}.h5"
        ), 
        monitor='val_sparse_categorical_accuracy',
        mode='auto',
        save_weights_only=True,
        save_best_only=True
    )

    # Train the model
    print("Training", name, "model")
    history = model.fit(
        x=train_generator,
        y=None,
        epochs=150,
        callbacks=[cp_callback, reduce_lr, early_stop],
        validation_data=val_generator,
        max_queue_size=1,
    )    
    histories.append(history)

# Get best iteration of each model based on validation accuracy performance
best_iters = [
    np.argmax(h.history["val_sparse_categorical_accuracy"]) for h in histories
]

# Plot the validation performances (loss and accuracy)
fig = plt.figure()
plt.title("Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
for history, name, best_iter in zip(histories, names, best_iters):
    plt.plot(history.history["val_loss"], label=name)
    plt.scatter([best_iter], [history.history["val_loss"][best_iter]])
plt.legend()
plt.savefig(os.path.join(SAVE_DIR, "validation_loss.png"), bbox_inches="tight")
plt.close()

fig = plt.figure()
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Categorical Accuracy")
for history, name, best_iter in zip(histories, names, best_iters):
    val_acc = history.history["val_sparse_categorical_accuracy"]
    plt.plot(val_acc, label=name)
    plt.scatter(
        [best_iter],
        [val_acc[best_iter]]
    )
plt.legend()
plt.savefig(os.path.join(SAVE_DIR, "validation_acc.png"), bbox_inches="tight")
plt.close()

# Evaluate the performance of each model and compare
performances = []
for model, best_iter, name in zip(models, best_iters, names):
    # Load the weights for this model with the best accuracy
    model.load_weights(os.path.join(
            cp_dir,
            name + "." + str(best_iter + 1).rjust(2, "0") + ".h5"
        )
    )

    # Prime the GPU, otherwise the time is unreliable
    test_preds = model.predict(test_images, batch_size=64, verbose=0)
    
    # Get predictions and measure performance time
    # Take the average over 5 iterations - this exercise assumes we care to
    #   some degree about the trade-off between inference time and accuracy
    start_time = time.time()
    for i in range(5):
        test_preds = model.predict(test_images, batch_size=64, verbose=0)
    elapsed_time = time.time() - start_time
    elapsed_time /= 5

    # Get time per batch
    elapsed_time_per_batch = elapsed_time / np.ceil(len(test_images)/64)
    
    # Create a confusion matrix for this model
    test_preds_argmax = np.argmax(test_preds, axis=1)
    cm = metrics.confusion_matrix(
        test_labels, test_preds_argmax, normalize='pred'
    )
    plt.figure("CM")
    sns.heatmap(
        cm,
        annot=True,
        cmap='Blues',
        fmt='.1%',
        xticklabels=classes,
        yticklabels=classes,
        cbar=False
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(
        os.path.join(SAVE_DIR, name + "_cm.png"),
        bbox_inches="tight"
    )
    plt.close()

    # Get the accuracy by category
    avg_accuracy=np.mean(np.diagonal(cm))
    
    # Get the test loss
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    loss = loss(test_labels, test_preds).numpy()
    
    # Get Averaged One-vs-Rest ROC (micro-averaged, but since the dataset
    #   is perfectly balanced there's no difference between micro- and
    #   macro-averaged)
    test_labels_onehot = tf.keras.utils.to_categorical(test_labels)
    fpr, tpr, _ = roc_curve(test_labels_onehot.ravel(), test_preds.ravel())
    
    # Package up all the performance data to be plotted outside the loop
    performances.append({
        "avg_accuracy": avg_accuracy,
        "loss": loss,
        "elapsed_time_per_batch": elapsed_time_per_batch,
        "fpr": fpr,
        "tpr":tpr
    })
    
# Compare the Receiver Operating Characteristic Curves between the models
plt.figure("ROCs")
for performance, name in zip(performances, names):
    plt.plot(performance["fpr"], performance["tpr"], label=name)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (One vs Rest)")
plt.legend()
plt.ylim(bottom=0.5)
plt.xlim(right=0.5)
plt.savefig(os.path.join(SAVE_DIR, "ROCs.png"), bbox_inches="tight")
plt.close()

# Create a table of the performance of each method
plt.figure("table")
plt.figure(figsize=(6,1))
ax = plt.gca()
ax.set_axis_off()
data = []
for perf in performances:
    data.append([
        "{0:.2%}".format(perf["avg_accuracy"]),
        "{0:.2f}".format(perf["avg_accuracy"] / (perf["elapsed_time_per_batch"] * 1000)),
        "{0:.2f}".format(perf["loss"]),
        "{0:.2f}".format(perf["elapsed_time_per_batch"] * 1000)
    ])

# Add a table at the bottom of the axes
columns = ('Accuracy', 'Accuracy per ms', 'Loss', 'ms per batch of 64')
ax.table(cellText=data,
          rowLabels=names,
          colLabels=columns,
          loc='center',
          cellLoc='center',
          rasterized=True)
plt.savefig(
    os.path.join(SAVE_DIR, 'performance_table.png'),
    bbox_inches='tight',
    dpi=300
)
plt.close()
