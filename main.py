import os
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET

import tensorflow as tf
from tensorflow import keras

import keras_cv
from keras_cv import bounding_box
from keras_cv import visualization
import numpy as np

os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'


SPLIT_RATIO = 0.2
BATCH_SIZE = 1
LEARNING_RATE = 0.001
EPOCH = 5
GLOBAL_CLIPNORM = 10.0
MAX_BOXES = 100


def get_classes_from_xml(filename):
    with open(filename) as f:
        tree = ET.parse(f)
        root = tree.getroot()

        classes = set()
        for obj in root.findall('object'):
            name = obj.find('name').text
            classes.add(name)
        return classes


class_ids = set()
path = 'data/karp/test/'
for filename in os.listdir(path):
    if filename.endswith('.xml'):
        class_ids.update(get_classes_from_xml(f'{path}{filename}'))
class_mapping = dict(zip(range(len(class_ids)), class_ids))

path_images = 'data/karp/test/'
path_annot = 'data/karp/test/'

# Get all XML file paths in path_annot and sort them
xml_files = sorted(
    [
        os.path.join(path_annot, file_name)
        for file_name in os.listdir(path_annot)
        if file_name.endswith(".xml")
    ]
)

# Get all JPEG image file paths in path_images and sort them
jpg_files = sorted(
    [
        os.path.join(path_images, file_name)
        for file_name in os.listdir(path_images)
        if file_name.endswith(".jpg")
    ]
)


def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_name = root.find("filename").text
    image_path = os.path.join(path_images, image_name)

    boxes = []
    classes = []
    for obj in root.iter("object"):
        cls = obj.find("name").text
        classes.append(cls)

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])

    class_ids = [
        list(class_mapping.keys())[list(class_mapping.values()).index(cls)]
        for cls in classes
    ]
    return image_path, boxes, class_ids


image_paths = []
bbox = []
classes = []
for xml_file in tqdm(xml_files):
    image_path, boxes, class_ids = parse_annotation(xml_file)

    # Pad bounding boxes and classes to have MAX_BOXES slots
    padded_boxes = np.zeros((MAX_BOXES, 4), dtype=np.float32)
    padded_classes = np.zeros((MAX_BOXES,), dtype=np.float32)

    # Copy the actual bounding box data into the padded arrays
    num_boxes = min(len(boxes), MAX_BOXES)
    padded_boxes[:num_boxes] = boxes[:num_boxes]
    padded_classes[:num_boxes] = class_ids[:num_boxes]

    image_paths.append(image_path)
    bbox.append(padded_boxes)
    classes.append(padded_classes)

image_paths = tf.constant(image_paths)
bbox = tf.constant(bbox)
classes = tf.constant(classes)

data = tf.data.Dataset.from_tensor_slices((image_paths, classes, bbox))

num_val = int(len(xml_files) * 0.5)
val_data = data.take(num_val)
train_data = data.skip(num_val)


TARGET_SIZE = (640, 640)  # Adjust to the expected YOLO model input size

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    # Resize the image to match the model's expected input size
    image = tf.image.resize(image, TARGET_SIZE)
    return image / 255.0  # Normalize the image if required by the model


def pad_bounding_boxes(bounding_boxes, max_boxes=100):
    """Pads bounding boxes and classes up to `max_boxes` using `tf.pad`."""
    num_boxes = tf.shape(bounding_boxes["boxes"])[0]
    padding_amount = max_boxes - num_boxes

    # Pad or truncate boxes and classes as needed
    boxes = tf.cond(
        padding_amount > 0,
        lambda: tf.pad(bounding_boxes["boxes"], [[0, padding_amount], [0, 0]]),
        lambda: bounding_boxes["boxes"][:max_boxes]  # Truncate if exceeding max_boxes
    )

    classes = tf.cond(
        padding_amount > 0,
        lambda: tf.pad(bounding_boxes["classes"], [[0, padding_amount]], constant_values=-1),
        lambda: bounding_boxes["classes"][:max_boxes]  # Truncate if exceeding max_boxes
    )

    return {"boxes": boxes, "classes": classes}


def load_dataset(image_path, classes, bbox):
    # Read and resize image
    image = load_image(image_path)

    # Cast and pad bounding boxes
    bounding_boxes = {
        "classes": tf.cast(classes, dtype=tf.float32),
        "boxes": bbox,
    }
    bounding_boxes = pad_bounding_boxes(bounding_boxes)

    # Return the image and bounding box dictionary in the required format
    return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}


# augmenter = keras.Sequential(
#     layers=[
#         keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="xyxy"),
#         keras_cv.layers.RandomShear(
#             x_factor=0, y_factor=0, bounding_box_format="xyxy"
#         ),
#         keras_cv.layers.JitteredResize(
#             target_size=(640, 640), scale_factor=(0.9, 1.1), bounding_box_format="xyxy"
#         ),
#     ]
# )

train_ds = train_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(BATCH_SIZE * 4)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True)
# train_ds = train_ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)


# def debug_shapes(data):
#     images = data["images"]
#     bounding_boxes = data["bounding_boxes"]
#     tf.print("Image shape:", tf.shape(images))
#     tf.print("Bounding boxes shape:", tf.shape(bounding_boxes["boxes"]))
#     tf.print("Classes shape:", tf.shape(bounding_boxes["classes"]))
#     return data  # Return the data unchanged so it can continue in the pipeline
#
#
# train_ds = train_ds.map(debug_shapes, num_parallel_calls=tf.data.AUTOTUNE)

resizing = keras_cv.layers.JitteredResize(
    target_size=(640, 640),
    scale_factor=(0.9, 1.1),
    bounding_box_format="xyxy",
)

val_ds = val_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.shuffle(BATCH_SIZE * 4)
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=True)
val_ds = val_ds.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)


def dict_to_tuple(inputs):
    return inputs["images"], inputs["bounding_boxes"]


train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

backbone = keras_cv.models.YOLOV8Backbone.from_preset(
    "yolo_v8_s_backbone_coco"
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


class EvaluateCOCOMetricsCallback(keras.callbacks.Callback):
    def __init__(self, data, save_path):
        super().__init__()
        self.data = data
        self.metrics = keras_cv.metrics.BoxCOCOMetrics(
            bounding_box_format="xyxy",
            evaluate_freq=1e9,
        )

        self.save_path = save_path
        self.best_map = -1.0

    def on_epoch_end(self, epoch, logs):
        self.metrics.reset_state()
        for batch in self.data:
            images, y_true = batch[0], batch[1]
            y_pred = self.model.predict(images, verbose=0)
            self.metrics.update_state(y_true, y_pred)

        metrics = self.metrics.result(force=True)
        logs.update(metrics)

        current_map = metrics["MaP"]
        if current_map > self.best_map:
            self.best_map = current_map
            self.model.save(self.save_path)  # Save the model when mAP improves

        return logs

for batch in train_ds.take(1):
    images, bounding_boxes = batch
    tf.print("Image shape:", tf.shape(images))
    tf.print("Bounding boxes:", bounding_boxes)
    # Run a single forward pass
    yolo(images)

yolo.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3,
    callbacks=[EvaluateCOCOMetricsCallback(val_ds, "model.h5")],
)
