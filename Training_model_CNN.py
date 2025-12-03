import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from Architecture import CusModel
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.metrics import F1Score
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight


#Uploading the dataset
base_dir = "Reorganized_Dataset_parula"
train_dir = os.path.join(base_dir, "Train")
val_dir = os.path.join(base_dir, "Val")
test_dir = os.path.join(base_dir, "Test")

#Choosing the parameters for the image
img_height, img_width = 224, 224
batch_size = 32

train_ds = image_dataset_from_directory(
    train_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True,
    label_mode="categorical"
)
val_ds = image_dataset_from_directory(
    val_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="categorical"
)

test_ds = image_dataset_from_directory(
    test_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False,
    label_mode="categorical"
)

class_names = test_ds.class_names
normalization_layer = Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE #Accelerating the process of training
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = CusModel(img_height, img_width) #model

dummy_x = tf.zeros((1, img_height, img_width, 3))
model(dummy_x) #it needs to show the size of layers for the summary

#Compilation
model.compile(optimizer='AdamW',
              loss='categorical_crossentropy',
              metrics=['accuracy',F1Score(average='macro',name='F1')])

model.summary()

#saving the best model
checkpoint_callback = ModelCheckpoint(
    filepath="best_f1_model.weights.h5",
    monitor='val_F1',
    mode='max',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

print("Computing class weights...")
y_train = []
for x, y in train_ds:
    y_train.extend(np.argmax(y.numpy(), axis=1))

y_train = np.array(y_train)

#Class weights to decrease the risk of memorizing of dataset by model
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(weights))
print(f"Class Weights: {class_weight_dict}")

#Training
history = model.fit(
    train_ds,
    epochs=20,
    validation_data=val_ds,
    callbacks=[checkpoint_callback],
    class_weight=class_weight_dict
)

#showing the best model
print("Loading best weights for evaluation...")
model.load_weights("best_f1_model.weights.h5")

#showing the results
test_loss, test_acc, test_f1 = model.evaluate(test_ds)
print(f"\n✅ Test accuracy: {test_acc:.3f}")
print(f"✅ Test F1 Score: {test_f1:.3f}")

# Confusion Matrix
y_true_one_hot = np.concatenate([y for x, y in test_ds], axis=0)
y_true = np.argmax(y_true_one_hot, axis=1)

y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_true, y_pred)
cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', values_format='d', ax=plt.gca(), colorbar=False)
plt.title("Confusion Matrix (Counts)")

plt.subplot(1, 2, 2)
disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm,display_labels=class_names)
disp_norm.plot(cmap='Blues', values_format='.2f', ax=plt.gca(), colorbar=False)
plt.title("Confusion Matrix (Normalized %)")
plt.tight_layout()
plt.show()

model.save_weights("ultrasound_cnn.weights.h5")
