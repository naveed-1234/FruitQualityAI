import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset_fruits360')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 3

# Update fruit list according to what your friend is collecting
FRUIT_CLASSES = [
    'Apple', 'Banana', 'Kiwi', 'Mango', 'Orange', 
    'Papaya', 'Peach', 'Pear', 'Pineapple', 'Strawberry'
]
QUALITY_CLASSES = ['Fresh', 'Overripe', 'Rotten']