import os
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET

import tensorflow as tf
from tensorflow import keras

import keras_cv
from keras_cv import bounding_box
from keras_cv import visualization

from CocometricCallback import EvaluateCOCOMetricsCallback

SPLIT_RATIO = 0.2
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCH = 5
GLOBAL_CLIPNORM = 10.0

labels = ['Blobs', 'Cracking-warping', 'Spaghetti', 'Stringging', 'Under Extrusion']
class_mapping = dict(zip(range(len(labels)), labels))

def parse_file(txtfile):
    file_name = os.path.splitext(os.path.basename(txtfile))[0]
    head, tail = os.path.split(txtfile)
    elements = head.split(os.path.sep)[:2]
    image_dir = os.path.join(elements[0], 'images', file_name + '.jpg')
    with open(txtfile, 'r') as file:
        lines = file.readlines()
        boxes = []
        classes = []
        for line in lines:
            elements = line.strip().split(' ')
            class_id = int(elements[0])
            classes.append(class_id)

            x = float(elements[1])
            y = float(elements[2])
            width = float(elements[3])
            height = float(elements[4])
            boxes.append([x, y, width, height])
    return image_dir, classes, boxes

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image


def load_dataset(image_path, classes, bbox):
    # Read Image
    image = load_image(image_path)
    bounding_boxes = {
        "classes": tf.cast(classes, dtype=tf.float32),
        "boxes": bbox,
    }
    return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}



train_annot_path = "Dataset_Combined/train/labels"

train_annot_files = sorted([
    os.path.join(train_annot_path, file_name)
    for file_name in os.listdir(train_annot_path)
    if file_name.endswith(".txt")
])
valid_annot_path = "Dataset_Combined/valid/labels"

valid_annot_files = sorted([
    os.path.join(valid_annot_path, file_name)
    for file_name in os.listdir(valid_annot_path)
    if file_name.endswith(".txt")
])

test_annot_path = "Dataset_Combined/test/labels"

test_annot_files = sorted([
    os.path.join(test_annot_path, file_name)
    for file_name in os.listdir(test_annot_path)
    if file_name.endswith(".txt")
])

train_image_paths = []
train_bbox = []
train_classes = []
for txtfile in tqdm(train_annot_files):
    image_path, boxes, class_ids = parse_file(txtfile)
    train_image_paths.append(image_path)
    train_bbox.append(boxes)
    train_classes.append(class_ids)

train_bbox = tf.ragged.constant(train_bbox)
train_classes = tf.ragged.constant(train_classes)
train_image_paths = tf.ragged.constant(train_image_paths)

train_data = tf.data.Dataset.from_tensor_slices((train_image_paths, train_classes, train_bbox))

train_ds = train_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(BATCH_SIZE * 4)
train_ds = train_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)


valid_image_paths = []
valid_bbox = []
valid_classes = []
for txtfile in tqdm(valid_annot_files):
    image_path, boxes, class_ids = parse_file(txtfile)
    valid_image_paths.append(image_path)
    valid_bbox.append(boxes)
    valid_classes.append(class_ids)

valid_bbox = tf.ragged.constant(valid_bbox)
valid_classes = tf.ragged.constant(valid_classes)
valid_image_paths = tf.ragged.constant(valid_image_paths)

valid_data = tf.data.Dataset.from_tensor_slices((valid_image_paths, valid_classes, valid_bbox))

valid_ds = valid_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
valid_ds = valid_ds.shuffle(BATCH_SIZE * 4)
valid_ds = valid_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)

test_image_paths = []
test_bbox = []
test_classes = []
for txtfile in tqdm(test_annot_files):
    image_path, boxes, class_ids = parse_file(txtfile)
    test_image_paths.append(image_path)
    test_bbox.append(boxes)
    test_classes.append(class_ids)

test_bbox = tf.ragged.constant(test_bbox)
test_classes = tf.ragged.constant(test_classes)
test_image_paths = tf.ragged.constant(test_image_paths)

test_data = tf.data.Dataset.from_tensor_slices((test_image_paths, test_classes, test_bbox))

test_ds = test_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.shuffle(BATCH_SIZE * 4)
test_ds = test_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)

backbone = keras_cv.models.YOLOV8Backbone.from_preset(
    "yolo_v8_s_backbone"  # We will use yolov8 small backbone with coco weights
)

yolo = keras_cv.models.YOLOV8Detector(
    num_classes=len(class_mapping),
    bounding_box_format="xyxy",
    backbone=backbone,
    fpn_depth=1,
)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    global_clipnorm=GLOBAL_CLIPNORM,
)

yolo.compile(
    optimizer=optimizer, classification_loss="binary_crossentropy", box_loss="ciou"
)

yolo.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=30,
    callbacks=[EvaluateCOCOMetricsCallback(valid_ds, "model.h5")],
)
