import torch
import torch.nn as nn
from torchvision import models
from pipelineClass import MultimodalPipeline # Assuming the class is in this file


class MultimodalHousingClassifier(nn.Module):
    def __init__(self, tabular_input_dim, num_classes, cnn_output_dim=512, tabular_emb_dim=128, pretrained=True):
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

if __name__ == '__main__':
    pipeline = MultimodalPipeline(
        model_class=MultimodalHousingClassifier,
        csv_path='../Data/Full_preprocessed_detailed_house.csv',
        image_base_dir='../Data/',
        image_col='frontview_url',
        target_col='woningtype',
        numeric_cols=['opp_pand', 'oppervlakte', 'build_year'],
        categorical_cols=['build_type'],
        epochs=1,  # Increased for demonstration
        lr=1e-4,
        batch_size=32
    )

    pipeline.train()
    pipeline.evaluate()
