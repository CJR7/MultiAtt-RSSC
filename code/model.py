import torch
import torch.nn as nn
from transformers import ViTModel, CLIPTextModel

class VisionTextModel(nn.Module):
    def __init__(self, num_classes, text_output_dim=512, cross_attention_hidden_size=512, nlevels=5, attn_dropout=0.05, relu_dropout=0.1, res_dropout=0.1, out_dropout=0.0, num_heads=4):
        super(VisionTextModel, self).__init__()
        self.image_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32')
        
        self.nlevels = nlevels
        self.text_output_dim = text_output_dim  # Hidden size of CLIP text model
        self.cross_attention_hidden_size = cross_attention_hidden_size  # Hidden size for cross-attention
        
        # Projection layers to match hidden sizes
        self.image_proj = nn.Linear(768, cross_attention_hidden_size)
        self.back_proj = nn.Linear(cross_attention_hidden_size, 768)
        
        self.text_self_attention = nn.MultiheadAttention(embed_dim=text_output_dim, num_heads=num_heads, dropout=attn_dropout)
        self.text_cross_attention = nn.MultiheadAttention(embed_dim=cross_attention_hidden_size, num_heads=num_heads, dropout=attn_dropout)
        self.image_cross_attention = nn.MultiheadAttention(embed_dim=cross_attention_hidden_size, num_heads=num_heads, dropout=attn_dropout)
        
        self.mlp = nn.Sequential(
            nn.Linear(cross_attention_hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(relu_dropout),
            nn.Linear(1024, cross_attention_hidden_size),
            nn.Dropout(res_dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(768 + cross_attention_hidden_size, num_classes),
            nn.Dropout(out_dropout)
        )

        # Layer Normalization layers
        self.text_layer_norm = nn.LayerNorm(text_output_dim)
        self.image_layer_norm = nn.LayerNorm(768)

    def forward(self, image, text):
        image_repr = self.image_encoder(image).last_hidden_state  # (B, P, 768)
        text_repr = self.text_encoder(text).last_hidden_state    # (B, S, text_output_dim)
        
        # Project image representation to match cross-attention hidden size
        image_repr_proj = self.image_proj(image_repr)  # (B, P, cross_attention_hidden_size)
        
        for _ in range(self.nlevels):
            # Self-attention on text representation
            text_self_att, _ = self.text_self_attention(text_repr, text_repr, text_repr)  # (B, S, text_output_dim)
            
            # Cross-attention: Text queries, Image keys and values
            enhanced_text_repr, _ = self.text_cross_attention(text_self_att.transpose(0, 1), image_repr_proj.transpose(0, 1), image_repr_proj.transpose(0, 1))  # (S, B, cross_attention_hidden_size)
            enhanced_text_repr = enhanced_text_repr.transpose(0, 1)  # (B, S, cross_attention_hidden_size)
            text_repr = self.text_layer_norm(text_self_att + enhanced_text_repr)
            
            # MLP on enhanced text representation
            enhanced_text_repr_mlp = self.mlp(text_repr)  # (B, S, cross_attention_hidden_size)
            
            # Cross-attention: Image queries, enhanced text keys and values
            enhanced_image_repr, _ = self.image_cross_attention(image_repr_proj.transpose(0, 1), enhanced_text_repr_mlp.transpose(0, 1), enhanced_text_repr_mlp.transpose(0, 1))  # (P, B, cross_attention_hidden_size)
            enhanced_image_repr = enhanced_image_repr.transpose(0, 1)  # (B, P, cross_attention_hidden_size)
            enhanced_image_repr = self.back_proj(enhanced_image_repr)  # Project back to original size (B, P, 768)
            image_repr = self.image_layer_norm(image_repr + enhanced_image_repr)

        # Pooling representations
        text_repr_pooled = text_repr.mean(dim=1)  # (B, text_output_dim)
        image_repr_pooled = image_repr.mean(dim=1)  # (B, 768)
        final_repr = torch.cat((image_repr_pooled, text_repr_pooled), dim=1)  # (B, 768 + text_output_dim)
        
        output = self.classifier(final_repr)  # (B, num_classes)
        return output
