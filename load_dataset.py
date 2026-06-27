from collections import defaultdict
import os
import cv2
import numpy as np

train_set = 'dataset/Train'
test_set = 'dataset/Test'
meta_set = 'dataset/Meta'

IMG_HEIGHT = 32
IMG_WIDTH = 32
channels = 3

NUM_CATEGORIES = len(os.listdir(train_set))

classes = { 
    0:'Speed limit (20km/h)',
    1:'Speed limit (30km/h)', 
    2:'Speed limit (50km/h)',            
    3:'Speed limit (60km/h)',         
    4:'Speed limit (70km/h)', 
    5:'Speed limit (80km/h)', 
    6:'End of speed limit (80km/h)', 
    7:'Speed limit (100km/h)', 
    8:'Speed limit (120km/h)', 
    9:'No passing', 
    10:'No passing veh over 3.5 tons', 
    11:'Right-of-way at intersection', 
    12:'Priority road', 
    13:'Yield', 
    14:'Stop', 
    15:'No vehicles',
    16:'Veh > 3.5 tons prohibited', 
    17:'No entry', 
    18:'General caution', 
    19:'Dangerous curve left', 
    20:'Dangerous curve right', 
    21:'Double curve', 
    22:'Bumpy road', 
    23:'Slippery road', 
    24:'Road narrows on the right', 
    25:'Road work', 
    26:'Traffic signals', 
    27:'Pedestrians', 
    28:'Children crossing', 
    29:'Bicycles crossing', 
    30:'Beware of ice/snow',
    31:'Wild animals crossing', 
    32:'End speed + passing limits', 
    33:'Turn right ahead', 
    34:'Turn left ahead', 
    35:'Ahead only', 
    36:'Go straight or right', 
    37:'Go straight or left', 
    38:'Keep right', 
    39:'Keep left',
    40:'Roundabout mandatory', 
    41:'End of no passing', 
    42:'End no passing veh > 3.5 tons'
}

def extract_track_id(filename):
    return filename.split('_')[0]

def load_test_data():
    image_data = []
    image_labels = []
    track_ids = []

    for i in range(NUM_CATEGORIES):
        path = os.path.join(train_set, str(i))
        print(f"Loading images from: {path}")
        for img_file in os.listdir(path):
            img_path = os.path.join(path, img_file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            image_data.append(img)
            image_labels.append(i)
            track_ids.append(extract_track_id(img_file))

    image_data = np.array(image_data, dtype=np.float32)
    image_labels = np.array(image_labels, dtype=np.int32)
    track_ids = np.array(track_ids)

    image_data = image_data.transpose(0, 3, 1, 2) / 255.0

    train_idx, val_idx = stratified_group_split(track_ids, image_labels, test_size=0.2, random_state=42)
    return (image_data[train_idx], image_labels[train_idx], image_data[val_idx], image_labels[val_idx])

def stratified_group_split(X, y, test_size=0.2, random_state=42):
    rng = np.random.RandomState(random_state)
    group_to_indices = defaultdict(list)

    for idx, group in enumerate(X):
        group_to_indices[group].append(idx)

    all_groups = list(group_to_indices.keys())
    rng.shuffle(all_groups)

    n_val = int(len(all_groups) * test_size)
    val_groups = set(all_groups[:n_val])

    train_idx = []
    val_idx = []

    for g, i in group_to_indices.items():
        if g in val_groups:
            val_idx.extend(i)
        else:
            train_idx.extend(i)

    return (np.array(train_idx, dtype=int), np.array(val_idx, dtype=int))