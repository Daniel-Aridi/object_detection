import os
import cv2 
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
from PIL import Image
import numpy as np

BASE_PATH = "C:/Users/Daniel/Desktop/objectDetection_py"



# Load pipeline configuration
PIPELINE_PATH = "{}/TensorFlow/models/my_ssd_resnet50_v1_fpn/pipeline.config".format(BASE_PATH)
configs = config_util.get_configs_from_pipeline_file(PIPELINE_PATH)

# Build the detection model
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
CHECKPOINT_PATH = "{}/TensorFlow/workspace/training_demo/exported-models/my_model_V1".format(BASE_PATH)
checkpoint = tf.train.Checkpoint(model=detection_model)
checkpoint.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-2')).expect_partial()

@tf.function
def detect_object(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


LABEL_MAP_PATH = "{}/TensorFlow/workspace/training_demo/annotations/label_map.pbtxt".format(BASE_PATH)
category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH)


# Function to load and preprocess an image
def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    image_np = np.array(image)
    return image_np




# Load and preprocess the image
IMAGE_PATH = "{}/TensorFlow/detection_script/detect_images/light1.jpg".format(BASE_PATH)
input_image = load_and_preprocess_image(IMAGE_PATH)

# Detect objects in the image
input_tensor = tf.convert_to_tensor(np.expand_dims(input_image, 0), dtype=tf.float32)
detections = detect_object(input_tensor)

num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
detections['num_detections'] = num_detections

# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

label_id_offset = 1
image_np_with_detections = input_image.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.3,
            agnostic_mode=False,
            line_thickness=8
        )


image_np_with_detections = cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB)
cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))





key = cv2.waitKey(0)
if key == ord('q'):
    cv2.destroyAllWindows()








