import pandas as pd
import numpy as np
import torch
from PIL import Image
import math
import os

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

class HW_Dataset(Dataset):
    
    def __init__(self, data_root, data_type, transform=None, max_seq_len=50):
        self.data_root = data_root
        self.transform = transform
        self.max_seq_len = max_seq_len
        self.data_type = data_type
        
        self.samples = self.load_dataset()
        self.char_to_int = self.build_vocab()
        
    def load_dataset(self):
        labelled_pairs = []
        with open(f"{self.data_root}/{self.data_type}.txt", 'r', encoding='utf-8') as file:
            file.seek(0)
            lines = file.readlines()
            for line in lines:
                line = line.split()
                labelled_pairs.append((line[0], line[1]))
        return labelled_pairs
    
    def build_vocab(self):
        unique_chars = []
        with open(f"{self.data_root}/hindi_vocab.txt", 'r', encoding='utf-8') as file:
            file.seek(0)
            lines = file.readlines()
            for line in lines:
                for char in line:
                    if char not in unique_chars:
                        unique_chars.append(char)
        char_to_int = {'<PAD>': 0, '<UNK>': 1}
        for i in range(len(unique_chars)):
            char_to_int[unique_chars[i]] = i + 2
        return char_to_int
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        
        # load image
        img = Image.open(os.path.join(self.data_root, img_path)).convert('L')
        if self.transform:
            img_tensor = self.transform(img)
        
        # Process text
        text_ids = [self.char_to_int.get(c, self.char_to_int['<UNK>']) for c in text]
        padded_text_ids = torch.zeros(self.max_seq_len, dtype=torch.long)
        padded_text_ids[:len(text_ids)] = torch.tensor(text_ids)
        
        return img_tensor, padded_text_ids


import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # We need layers to calculate the attention "energy"
        # This will score how important each RNN output state is.
        self.W_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V_a = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, rnn_outputs):
        # rnn_outputs shape: (Batch_size, Max_seq_len, Hidden_size)
        # Calculate the attention "energy" for each hidden state
        scores = torch.tanh(self.W_a(rnn_outputs))
        energy = self.V_a(scores)
        # Get the attention weights (alpha) by applying softmax
        alpha_weights = F.softmax(energy.squeeze(2), dim=1)
        # Calculate the context vector
        context_vector = torch.bmm(alpha_weights.unsqueeze(1), rnn_outputs)
        return context_vector.squeeze(1)

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Use an LSTM to process the sequence of characters
        self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        # Add an attention layer
        rnn_output_size = hidden_size * 2
        self.attention = Attention(rnn_output_size)
        self.fc = nn.Linear(rnn_output_size, output_size)
        
    def forward(self, text_ids):
        # text_ids shape: (Batch_size, Max_seq_len)
        embedded = self.embedding(text_ids)
        # Pass through RNN/LSTM
        output, (hidden, _) = self.rnn(embedded)
        # Pass all RNN outputs to the attention layer
        context_vector = self.attention(output)
        condition = self.fc(context_vector)
        return condition

class Generator(nn.Module):
    def __init__(self, z_dim, condition_dim, img_channels, img_size_h, img_size_w):
        super().__init__()
        self.img_size_h = img_size_h
        self.img_size_w = img_size_w
        self.img_channels = img_channels
        
        # We start by projecting and reshaping the combined input
        # Input: z_dim + condition_dim
        self.fc = nn.Sequential(
            nn.Linear(z_dim + condition_dim, 1024 * (img_size_h // 16) * (img_size_w // 16)),
            nn.BatchNorm1d(1024 * (img_size_h // 16) * (img_size_w // 16)),
            nn.ReLU()
        )
        
        # Now, we upsample using ConvTranspose2d
        # We'll go from (4x16) -> (8x32) -> (16x64) -> (32x128) -> (64x256)
        self.gen = nn.Sequential(
            # Input: (1024, 4, 16)
            self._block(1024, 512, 4, 2, 1),  # -> (512, 8, 32)
            self._block(512, 256, 4, 2, 1),   # -> (256, 16, 64)
            self._block(256, 128, 4, 2, 1),   # -> (128, 32, 128)
            
            # Final layer to get to the target size and channels
            nn.ConvTranspose2d(
                128, img_channels, kernel_size=4, stride=2, padding=1
            ),
            # Output: (img_channels, 64, 256)
            nn.Tanh() # Normalize output to [-1, 1], matching data normalization
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, z, condition):
        # z shape: (N, z_dim)
        # condition shape: (N, condition_dim)
        
        # Combine noise and condition
        combined_input = torch.cat([z, condition], dim=1) # (N, z_dim + condition_dim)
        
        # Project and reshape
        x = self.fc(combined_input)
        # Reshape to (N, 1024, H/16, W/16) -> (N, 1024, 4, 16)
        x = x.view(-1, 1024, self.img_size_h // 16, self.img_size_w // 16)
        
        # Pass through the generator blocks
        return self.gen(x)
    
class Discriminator(nn.Module):
    def __init__(self, condition_dim, img_channels, img_size_h, img_size_w):
        super().__init__()
        
        # CNN blocks to process the image
        # Input: (img_channels, 64, 256)
        self.disc = nn.Sequential(
            # -> (128, 32, 128)
            self._block(img_channels, 128, 4, 2, 1, use_norm=False), 
            # -> (256, 16, 64)
            self._block(128, 256, 4, 2, 1),
            # -> (512, 8, 32)
            self._block(256, 512, 4, 2, 1),
            # -> (1024, 4, 16)
            self._block(512, 1024, 4, 2, 1),
        )
        
        # Flatten and combine with condition
        # Output of disc: (N, 1024, 4, 16)
        # Flattened size: 1024 * 4 * 16 = 65536
        self.fc = nn.Sequential(
            nn.Linear(1024 * (img_size_h // 16) * (img_size_w // 16) + condition_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1)
            # No Sigmoid here!
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding, use_norm=False):
        layers = [
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            )
        ]
        if use_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def forward(self, x, condition):
        # x shape: (N, C, H, W)
        # condition shape: (N, condition_dim)
        
        x = self.disc(x) # (N, 1024, 4, 16)
        
        # Flatten and concatenate condition
        x_flat = x.view(x.shape[0], -1) # (N, 65536)
        combined = torch.cat([x_flat, condition], dim=1) # (N, 65536 + condition_dim)
        
        # Classify
        return self.fc(combined)

from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import functional as TF
import torch.nn.functional as F

class FixedHeightResize:
    """Resizes an image to a fixed height while preserving the aspect ratio."""
    def __init__(self, target_height=64):
        self.target_height = target_height

    def __call__(self, img):
        # Get original dimensions (PIL image returns width, height)
        original_width, original_height = img.size
        # Calculate the new width to maintain the aspect ratio
        aspect_ratio = original_width / original_height
        new_width = math.ceil(aspect_ratio * self.target_height)
        # Resize the image
        resized_img = img.resize((new_width, self.target_height), Image.BICUBIC)
        return resized_img

class PadToWidth:
    """Pads an image (PIL or Tensor) to a fixed width while maintaining height."""
    def __init__(self, target_width, fill_color=255): 
        # For standard handwriting data on a white background, 255 (white) is best.
        self.target_width = target_width
        self.fill_color = fill_color 

    def __call__(self, img):
        if not isinstance(img, Image.Image):
             # Assumes we are operating on a PIL Image before ToTensor()
             raise TypeError("Input must be a PIL Image.")

        current_width = img.width
        
        if current_width >= self.target_width:
            # If the image is already wide enough (or too wide), we just center-crop it.
            # This handles outliers, though you should choose max_width carefully.
            return TF.center_crop(img, (img.height, self.target_width))
        
        # Calculate padding needed (only on the right)
        padding_needed = self.target_width - current_width
        
        # Pad with the background color (left, top, right, bottom)
        # We only pad on the right (right padding = padding_needed)
        padding = (0, 0, padding_needed, 0) 
        
        return TF.pad(img, padding, fill=self.fill_color)


def calculate_gradient_penalty(critic, real_samples, fake_samples, condition, DEVICE, LAMBDA_GP):
    """Calculates the gradient penalty for WGAN-GP."""

    # Randomly sample interpolation points between real and fake
    alpha = torch.rand((real_samples.size(0), 1, 1, 1)).to(DEVICE)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True).to(DEVICE)

    # Critic scores on interpolates
    interpolates_out = critic(interpolates, condition)

    # Calculate gradients of the critic's output w.r.t. the input interpolates
    gradients = torch.autograd.grad(
        outputs=interpolates_out,
        inputs=interpolates,
        grad_outputs=torch.ones_like(interpolates_out),
        create_graph=True,  # Needed for second-order derivatives in the training loop
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Calculate penalty
    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = LAMBDA_GP * ((gradient_norm - 1) ** 2).mean()

    return gradient_penalty