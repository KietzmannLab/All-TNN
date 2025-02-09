import os
import json

# Define the paths for the JSON files
current_directory = os.path.dirname(os.path.abspath(__file__))
coco_id_path = os.path.join(current_directory, 'util_data/coco_id_to_name_mapping.json')
ecoset_id_path = os.path.join(current_directory, 'util_data/ecoset_id_to_coco_id_mapping.json')
ecoset_id_to_name_mapping_path = os.path.join(current_directory, 'util_data/ecoset_id_to_name_mapping.json')

# Load and process COCO ID to Name Mapping
with open(coco_id_path, 'r') as file:
    coco_id_to_name = json.load(file)
COCO_ID_NAME_MAP = {int(k): v for k, v in coco_id_to_name.items()}
COCO_NAME_ID_MAP = {v: int(k) for k, v in COCO_ID_NAME_MAP.items()}

# Load ecoset ID to Name Mapping
with open(ecoset_id_to_name_mapping_path, 'r') as file:
    ecoset_ID2Name = json.load(file)

# Load ecoset ID to COCO ID Mapping
with open(ecoset_id_path, 'r') as file:
    ecoset_ID2COCO_ID = json.load(file)

# Define category names used in Behaviour Experiment
CATEGORY_NAMES = [
    'bus', 'airplane', 'train', 'motorcycle', 'bear', 'elephant',
    'giraffe', 'zebra', 'cat', 'kite', 'pizza', 'broccoli',
    'laptop', 'refrigerator', 'scissors', 'toilet'
]

# Define clustered category names
CLUSTERED_CATEGORY_NAMES = [
    "train", "bus", "pizza", "broccoli", "toilet", "motorcycle",
    "zebra", "laptop", "kite", "airplane", "cat", "bear", 
    "elephant", "giraffe", "refrigerator", "scissors"
]

# Get COCO IDs for the defined category names
BEHAVIOUR_CATEGORIES_ORDERED_COCO_ID = [COCO_NAME_ID_MAP[name] for name in CATEGORY_NAMES]