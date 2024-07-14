import os
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split


class ImageTextDataset(Dataset):
    def __init__(self, data_path=None, image_paths=None, text_paths=None, labels=None, split=None):
        self.data_path = data_path
        self.split = split
        self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        self.train = None
        self.val = None
        self.test = None
        self.class_names = []

        if self.data_path:
            self.image_paths = []
            self.text_paths = []
            self.labels = []
        else:
            self.image_paths = image_paths
            self.text_paths = text_paths
            self.labels = labels

        if self.split:
            if self.split.isdigit():
                self.train, self.val = self._load_train_data(ratio = int(self.split)/100)
            elif self.split == 'test':
                self.test = self._load_test_data()
            else:
                raise ValueError(f"Unsupported mode: {self.split}")



    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        with open(self.text_paths[idx], 'r') as f:
            text = f.read().strip()
        label = self.labels[idx]
        
        # Preprocess the image (resize to 224x224)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        image = transform(image)
        
        # Tokenize and pad text
        inputs = self.processor(text=[text], images=image, return_tensors="pt", padding='max_length', truncation=True,do_rescale=False)
        
        return inputs['pixel_values'].squeeze(0), inputs['input_ids'].squeeze(0), label

    def _load_data(self):
        image_dir = os.path.join(self.data_path, 'image')
        text_dir = os.path.join(self.data_path, 'text')
        valid_extensions = ('.jpg', '.png' , '.tif')
        
        for class_idx, class_name in enumerate(sorted(os.listdir(image_dir))):
            class_image_dir = os.path.join(image_dir, class_name)
            class_text_dir = os.path.join(text_dir, class_name)
            if os.path.isdir(class_image_dir) and os.path.isdir(class_text_dir):
                self.class_names.append(class_name)
            
                for file_name in sorted(os.listdir(class_image_dir)):
                    if file_name.endswith(valid_extensions):
                        image_path = os.path.join(class_image_dir, file_name)
                        text_path = os.path.join(class_text_dir, os.path.splitext(file_name)[0] + '.txt')
                        if os.path.isfile(image_path) and os.path.isfile(text_path):
                            self.image_paths.append(image_path)
                            self.text_paths.append(text_path)
                            self.labels.append(class_idx)

    def _load_train_data(self,ratio = 0.8):
        self._load_data()
        image_train, image_val, text_train, text_val, labels_train, labels_val = train_test_split(
        self.image_paths, self.text_paths, self.labels, train_size=ratio, stratify=self.labels, random_state=42)

        train_dataset = ImageTextDataset(data_path=None, image_paths = image_train, text_paths = text_train, labels = labels_train,split = None )
        val_dataset = ImageTextDataset(data_path=None, image_paths = image_val, text_paths = text_val, labels = labels_val,split = None)
        return train_dataset, val_dataset

    def _load_test_data(self):
        self._load_data()

    def get_class_names(self):
        return self.class_names

    def get_num_classes(self):
        return len(set(self.labels))

if __name__ == '__main__':
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser(description='Test ImageTextDataset')
        parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
        parser.add_argument('--batch_size', type=int, default=4, metavar='N', help='Batch size (default: 4)')
        parser.add_argument('--mode', type=str, choices=['80', '50', '20', 'test'], required=True, help='Mode: train split ratio or test')

        return parser.parse_args()

    args = parse_args()

    if args.mode == 'test':
        # Create dataset and dataloader
        test_dataset = ImageTextDataset(data_path=args.data_path, split=args.mode)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
        print(len(test_dataset))
        print(test_dataset.get_class_names())

        # Iterate through the dataset
        for batch in test_dataloader:
            images, texts, labels = batch
            print(f'Images shape: {images.shape}')  # (batch_size, 3, 224, 224)
            print(f'Texts shape: {texts.shape}')  # (batch_size, max_sequence_length)
            print(f'Labels: {labels}')
            break  # Remove this break to iterate through the entire dataset
    
    else:
        # Create dataset and dataloader
        train = ImageTextDataset(data_path=args.data_path, split=args.mode)
        train_data, val_data = train.train, train.val

        print(len(train_data))
        print(train_data.get_class_names())#got a bug unsolved

        train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)

        # Iterate through the dataset
        for batch in train_dataloader:
            images, texts, labels = batch
            print(f'Images shape: {images.shape}')  # (batch_size, 3, 224, 224)
            print(f'Texts shape: {texts.shape}')  # (batch_size, max_sequence_length)
            print(f'Labels: {labels}')
            break  # Remove this break to iterate through the entire dataset




