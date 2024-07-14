import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from model import VisionTextModel
from transformers import CLIPProcessor

def process_dataset(root_dir, model_path, num_classes, device='cuda'):
    # Load the model
    model = VisionTextModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # CLIP processor for text
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    representations = []
    labels = []
    class_names = []

    # Iterate through class folders
    for class_name in sorted(os.listdir(os.path.join(root_dir, 'image'))):
        class_names.append(class_name)
        image_class_dir = os.path.join(root_dir, 'image', class_name)
        text_class_dir = os.path.join(root_dir, 'text', class_name)
        
        if not os.path.isdir(image_class_dir):
            continue

        # Process each image in the class folder
        for img_name in os.listdir(image_class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(image_class_dir, img_name)
                txt_path = os.path.join(text_class_dir, os.path.splitext(img_name)[0] + '.txt')

                # Load and process image
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)

                # Load and process text
                with open(txt_path, 'r') as f:
                    text = f.read().strip()
                text_inputs = processor(text=text, return_tensors="pt", padding='max_length', truncation=True, max_length=77)
                text_tensor = text_inputs['input_ids'].to(device)

                # Get final representation
                with torch.no_grad():
                    image_repr = model.image_encoder(image_tensor).last_hidden_state
                    text_repr = model.text_encoder(text_tensor).last_hidden_state

                    # Perform the same processing as in your forward method
                    for _ in range(model.nlevels):
                        text_self_att, _ = model.text_self_attention(text_repr, text_repr, text_repr)
                        image_repr_proj = model.image_proj(image_repr)
                        enhanced_text_repr, _ = model.text_cross_attention(text_self_att.transpose(0, 1), image_repr_proj.transpose(0, 1), image_repr_proj.transpose(0, 1))
                        enhanced_text_repr = enhanced_text_repr.transpose(0, 1)
                        text_repr = model.text_layer_norm(text_self_att + enhanced_text_repr)
                        enhanced_text_repr_mlp = model.mlp(text_repr)
                        enhanced_image_repr, _ = model.image_cross_attention(image_repr_proj.transpose(0, 1), enhanced_text_repr_mlp.transpose(0, 1), enhanced_text_repr_mlp.transpose(0, 1))
                        enhanced_image_repr = enhanced_image_repr.transpose(0, 1)
                        enhanced_image_repr = model.back_proj(enhanced_image_repr)
                        image_repr = model.image_layer_norm(image_repr + enhanced_image_repr)

                    # Pooling representations
                    text_repr_pooled = text_repr.mean(dim=1)  # (1, text_output_dim)
                    image_repr_pooled = image_repr.mean(dim=1)  # (1, 768)
                    final_repr = torch.cat((image_repr_pooled, text_repr_pooled), dim=1)  # (1, 768 + text_output_dim)

                representations.append(final_repr.cpu().numpy())
                labels.append(class_names.index(class_name))

    return np.vstack(representations), np.array(labels), class_names

def plot_tsne_3d(representations, labels, class_names):
    # Perform t-SNE
    tsne = TSNE(n_components=3, random_state=42)
    tsne_results = tsne.fit_transform(representations)

    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create a scatter plot
    scatter = ax.scatter(
        tsne_results[:, 0],
        tsne_results[:, 1],
        tsne_results[:, 2],
        c=labels,
        cmap='tab20'
    )

    # Add a color bar
    #plt.colorbar(scatter)

    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=class_name,
                                  markerfacecolor='none', markeredgecolor=plt.cm.tab20(i/len(class_names)),
                                  markersize=10, alpha=0.8)
                       for i, class_name in enumerate(class_names)]
    
    # Place the legend outside the plot
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.1, 0.5),
              title="Classes", ncol=2)  # Adjust ncol as needed
    

    ax.set_title('3D t-SNE visualization of image-text representations')
    ax.set_xlabel('t-SNE feature 1')
    ax.set_ylabel('t-SNE feature 2')
    ax.set_zlabel('t-SNE feature 3')

    plt.tight_layout()
    plt.show()


def plot_tsne(representations, labels, class_names):
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(representations)

    # Plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab20',edgecolors='face',
        facecolors='none',
        linewidths=1,
        alpha=0.8)
    #plt.colorbar(scatter)

    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=class_name,
                                  markerfacecolor='none', markeredgecolor=plt.cm.tab20(i/len(class_names)),
                                  markersize=10, alpha=0.8)
                       for i, class_name in enumerate(class_names)]
    
    # Place the legend outside the plot
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5),
               title="Classes", ncol=2)  # Adjust ncol as needed

    plt.title('t-SNE visualization of image-text representations')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.tight_layout()
    plt.show()

# Example usage
root_dir = '../dataset/AID-test20/'#path to dataset
model_path = '../model/best-AID80.pth'  #path to model
num_classes = 30

representations, labels, class_names = process_dataset(root_dir, model_path, num_classes)
plot_tsne(representations, labels, class_names)
