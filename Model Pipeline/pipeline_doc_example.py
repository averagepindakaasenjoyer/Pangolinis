import os
import torch
import torch.nn as nn
from torchvision import models
from pipelineClass import MultimodalPipeline # Assuming the class is in this file
from dotenv import load_dotenv

class MultimodalHousingClassifier(nn.Module):
    def __init__(self, tabular_input_dim, num_classes, cnn_output_dim=512, tabular_emb_dim=128, pretrained=True, useMask=False):
        super().__init__()
        
        # CNN Backbone (ResNet18)
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.cnn_output_dim = cnn_output_dim
        
        # MLP for tabular features
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, tabular_emb_dim),
            nn.ReLU(),
        )
        
        # Fusion and Classifier Head
        fusion_input_dim = cnn_output_dim + tabular_emb_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, image, tabular_data):
        cnn_features = self.cnn_backbone(image)
        cnn_features = cnn_features.view(-1, self.cnn_output_dim)
        
        tabular_features = self.tabular_mlp(tabular_data)
        
        combined = torch.cat([cnn_features, tabular_features], dim=1)
        
        output = self.classifier(combined)
        return output

class CNNWithTabular(nn.Module):
    def __init__(self, tabular_input_dim: int, num_classes: int, image_out_dim: int = 512, tabular_emb_dim: int = 32, useMask: bool = False):
        """
        Initializes the multimodal model combining a CNN for images and an MLP for tabular data.

        Args:
            tabular_input_dim (int): The number of features in the tabular data.
            num_classes (int): The number of output classes for classification.
            image_out_dim (int): The output dimension of the CNN branch before fusion. Default is 512.
            tabular_emb_dim (int): The output dimension of the tabular MLP branch before fusion. Default is 32.
            useMask (bool): If True, the CNN expects an image with 4 channels (RGB + Mask).
                            If False, it expects 3 channels (RGB).
        """
        super().__init__()

        self.useMask = useMask
        input_image_channels = 4 if self.useMask else 3 # Determine input channels based on useMask

        # CNN part (image)
        # Note: Your original CNN example uses 1 input channel. For typical images
        # loaded by your pipeline (RGB), it should be 3. If you add a mask, it becomes 4.
        self.cnn = nn.Sequential(
            nn.Conv2d(input_image_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # If input image is 224x224, output here is 64x112x112

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # Output here is 128x56x56

            nn.Flatten(),
            # Adjust the input dimension to the linear layer based on the image_size
            # (H, W) from the pipeline. For 224x224 input:
            # After 2 MaxPool2d layers (each halves dimensions): 224 -> 112 -> 56
            # So, the flattened size is 128 * 56 * 56
            nn.Linear(128 * (224 // 4) * (224 // 4), image_out_dim), nn.ReLU(), # (H // 4) * (W // 4)
            nn.Dropout(0.3)
        )
        self.image_out_dim = image_out_dim # Store this for fusion layer

        # Tabular part
        self.tabular_net = nn.Sequential(
            nn.Linear(tabular_input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, tabular_emb_dim), nn.ReLU() # Output dimension aligned with tabular_emb_dim
        )
        self.tabular_emb_dim = tabular_emb_dim # Store this for fusion layer

        # Final combined classifier
        self.final = nn.Sequential(
            nn.Linear(self.image_out_dim + self.tabular_emb_dim, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes) # Output dimension aligned with num_classes
        )

    def forward(self, image_input: torch.Tensor, tabular_input: torch.Tensor, mask_input: torch.Tensor = None) -> torch.Tensor:
        """
        Performs the forward pass of the multimodal model.

        Args:
            image_input (torch.Tensor): The input image tensor (e.g., [batch_size, channels, H, W]).
                                         Channels should be 3 (RGB) or 4 (RGB + Mask) depending on useMask.
            tabular_input (torch.Tensor): The input tabular data tensor (e.g., [batch_size, tabular_input_dim]).
            mask_input (torch.Tensor, optional): The input image mask tensor (e.g., [batch_size, 1, H, W]).
                                                 Only provided if self.useMask is True by the pipeline.

        Returns:
            torch.Tensor: The raw logits for classification (e.g., [batch_size, num_classes]).
        """
        # If useMask is True, concatenate the mask as an additional channel
        if self.useMask and mask_input is not None:
            # Ensure mask_input has a channel dimension of 1.
            # Your _rasterize_polygon returns a numpy array, HousingDataset unsqueezes it.
            # So, mask_input should be (N, 1, H, W).
            image_input = torch.cat((image_input, mask_input), dim=1)
        # else: image_input is already (N, 3, H, W)

        img_out = self.cnn(image_input)
        tab_out = self.tabular_net(tabular_input)

        x = torch.cat([img_out, tab_out], dim=1)
        return self.final(x)


class MultimodalHousingClassifierWithMask(nn.Module):
    def __init__(self, tabular_input_dim, num_classes, cnn_output_dim=512, tabular_emb_dim=128, pretrained=True, useMask=True):
        super().__init__()

        self.useMask = useMask
        
        # Determine the number of input channels for the CNN
        # RGB is 3 channels. If useMask is True, we add 1 channel for the mask.
        input_image_channels = 4 if self.useMask else 3 

        # CNN Backbone (ResNet18)
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Modify the first convolutional layer of ResNet if input channels are not 3
        if input_image_channels != 3:
            # Get the original first conv layer
            original_conv1 = resnet.conv1
            # Create a new conv layer with the desired input channels
            # Keep other parameters (output channels, kernel size, stride, padding, bias) the same
            resnet.conv1 = nn.Conv2d(input_image_channels, 
                                     original_conv1.out_channels, 
                                     kernel_size=original_conv1.kernel_size, 
                                     stride=original_conv1.stride, 
                                     padding=original_conv1.padding, 
                                     bias=original_conv1.bias)
            # Note: If pretrained=True, the new conv1 layer will be randomly initialized,
            # while the rest of the pretrained weights will be loaded. This is generally fine
            # as the first layer often adapts quickly during fine-tuning.

        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-1])
        # The output dimension of ResNet18 before the final pooling is 512.
        # This matches `cnn_output_dim`'s default, so no change needed here.
        self.cnn_output_dim = cnn_output_dim
        
        # MLP for tabular features
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, tabular_emb_dim),
            nn.ReLU(),
        )
        
        # Fusion and Classifier Head
        fusion_input_dim = cnn_output_dim + tabular_emb_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, image: torch.Tensor, tabular_data: torch.Tensor, mask_input: torch.Tensor = None) -> torch.Tensor:
        """
        Performs the forward pass of the multimodal model.

        Args:
            image (torch.Tensor): The input image tensor (e.g., [batch_size, 3, H, W]).
            tabular_data (torch.Tensor): The input tabular data tensor (e.g., [batch_size, tabular_input_dim]).
            mask_input (torch.Tensor, optional): The input image mask tensor (e.g., [batch_size, 1, H, W]).
                                                Only provided if self.useMask is True.

        Returns:
            torch.Tensor: The raw logits for classification (e.g., [batch_size, num_classes]).
        """
        # If useMask is True, concatenate the mask as an additional channel
        if self.useMask and mask_input is not None:
            # Ensure mask_input has a channel dimension of 1, and image is (N, 3, H, W).
            # The result will be (N, 4, H, W).
            image = torch.cat((image, mask_input), dim=1)
        
        cnn_features = self.cnn_backbone(image)
        # ResNet's backbone typically outputs (batch_size, 512, 1, 1) after average pooling.
        # We need to flatten it to (batch_size, 512).
        cnn_features = cnn_features.view(-1, self.cnn_output_dim) 
        
        tabular_features = self.tabular_mlp(tabular_data)
        
        combined = torch.cat([cnn_features, tabular_features], dim=1)
        
        output = self.classifier(combined)
        return output

if __name__ == '__main__':

    load_dotenv()
    BASE_DIR = os.getenv('FILE_PATH')
    if not BASE_DIR:
        raise ValueError("FILE_PATH environment variable not set. Please create a .env file and set it.")

    DATA_PATH = os.path.join(BASE_DIR, 'Full_preprocessed_detailed_house.csv')
    print(f"Base Directory: {BASE_DIR}")
    print(f"Data CSV Path: {DATA_PATH}")

    pipeline = MultimodalPipeline(
        model_class=MultimodalHousingClassifierWithMask,
        csv_path=DATA_PATH,
        image_base_dir=BASE_DIR,
        image_col='frontview_url',
        target_col='woningtype',
        numeric_cols=['opp_pand', 'oppervlakte', 'build_year'],
        categorical_cols=['build_type'],
        epochs=10,
        lr=1e-4,
        batch_size=32,
        useMask=True
        )

    pipeline.train()
    pipeline.evaluate()
