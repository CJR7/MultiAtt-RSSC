import os
import random
import shutil
from pathlib import Path

def create_sampled_dataset(original_dataset_path, sampled_dataset_path, sample_ratio=0.1):
    image_dir = os.path.join(original_dataset_path, 'image')
    text_dir = os.path.join(original_dataset_path, 'text')
    
    # Create the sampled dataset directory
    sampled_image_dir = os.path.join(sampled_dataset_path, 'image')
    sampled_text_dir = os.path.join(sampled_dataset_path, 'text')
    os.makedirs(sampled_image_dir, exist_ok=True)
    os.makedirs(sampled_text_dir, exist_ok=True)

    # Iterate over each class directory
    for class_name in sorted(os.listdir(image_dir)):
        class_image_dir = os.path.join(image_dir, class_name)
        class_text_dir = os.path.join(text_dir, class_name)
        
        if os.path.isdir(class_image_dir) and os.path.isdir(class_text_dir):
            # Create corresponding directories in the sampled dataset
            sampled_class_image_dir = os.path.join(sampled_image_dir, class_name)
            sampled_class_text_dir = os.path.join(sampled_text_dir, class_name)
            os.makedirs(sampled_class_image_dir, exist_ok=True)
            os.makedirs(sampled_class_text_dir, exist_ok=True)
            
            # Get all files in the class directory
            files = sorted(os.listdir(class_image_dir))
            valid_extensions = ('.jpg', '.png', '.tif')
            files = [f for f in files if f.endswith(valid_extensions)]
            
            # Sample given ratio of the files
            sample_size = int(len(files) * sample_ratio)
            sampled_files = random.sample(files, sample_size)
            
            # Move sampled files to the new dataset structure
            for file_name in sampled_files:
                image_path = os.path.join(class_image_dir, file_name)
                text_file_name = os.path.splitext(file_name)[0] + '.txt'
                text_path = os.path.join(class_text_dir, text_file_name)
                
                sampled_image_path = os.path.join(sampled_class_image_dir, file_name)
                sampled_text_path = os.path.join(sampled_class_text_dir, text_file_name)
                
                if os.path.isfile(image_path) and os.path.isfile(text_path):
                    shutil.move(image_path, sampled_image_path)
                    shutil.move(text_path, sampled_text_path)
                    print(f'Moved {image_path} and {text_path} to sampled dataset.')

if __name__ == '__main__':
    original_dataset_path = 'path/to/your/dataset' #save structure with ./minitest
    sampled_dataset_path = 'path/to/your/save_test_dataset'
    create_sampled_dataset(original_dataset_path, sampled_dataset_path, sample_ratio=0.1)
