import os
import torch
import torch.nn as nn
from torchvision import models
from pipelineClass import MultimodalPipeline
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# Model definition

class TabularModel(nn.Module):
    """
    MLP for tabular features matching `tabular_emb_dim`.
    """
    def __init__(self, input_dim, emb_dim=128, hidden_dim=256, dropout=0.3):
        super(TabularModel, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_tab):
        return self.mlp(x_tab)

class RandomForestTabular:
    """
    Wrapper for sklearn RandomForestClassifier for tabular fusion.
    """
    def __init__(self, input_dim, num_classes, **kwargs):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, **kwargs)
        self.num_classes = num_classes
        self.is_fitted = False

    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted = True

    def predict_proba(self, X):
        if not self.is_fitted:
            raise RuntimeError("RandomForestTabular must be fitted before calling predict_proba.")
        return self.model.predict_proba(X)

class XGBoostTabular:
    """
    Wrapper for XGBoost XGBClassifier for tabular fusion.
    """
    def __init__(self, input_dim, num_classes, **kwargs):
        self.model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss', **kwargs)
        self.num_classes = num_classes
        self.is_fitted = False

    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted = True

    def predict_proba(self, X):
        if not self.is_fitted:
            raise RuntimeError("XGBoostTabular must be fitted before calling predict_proba.")
        return self.model.predict_proba(X)
    

class ImageModel(nn.Module):
    """
    ResNet18 backbone producing a feature vector of size `cnn_output_dim`.
    """
    def __init__(self, cnn_output_dim=512, pretrained=True):
        super(ImageModel, self).__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)
        # drop the final fc layer and avgpool
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.cnn_output_dim = cnn_output_dim

    def forward(self, x_img):
        # x_img: [B,3,H,W]
        feat = self.backbone(x_img)               # [B, cnn_output_dim, 1, 1]
        return feat.view(feat.size(0), self.cnn_output_dim)

class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = models.resnet50(pretrained=True)

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Replace the final fully connected layer
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.classifier(x)
        return x

class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.base_model = models.resnet50(pretrained=pretrained)

        # Remove the final fully connected layer
        self.base_model.fc = nn.Identity()
        self.output_dim = 2048  # This is the output dimension of resnet50 without the fc layer

        # Optionally freeze weights
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.base_model(x)


class EarlyFusionModel(nn.Module):
    """
    Early fusion classifier:
    Args:
        tabular_input_dim (int): number of tabular features
        num_classes (int): number of output classes
        cnn_output_dim (int): image embedding size
        tabular_emb_dim (int): tabular embedding size
        dropout (float): dropout rate
        pretrained (bool): use imagenet weights
    """
    def __init__(self,
                 tabular_input_dim,
                 num_classes,
                 cnn_output_dim=512,
                 tabular_emb_dim=128,
                 dropout=0.5,
                 pretrained=True):
        super(EarlyFusionModel, self).__init__()
        self.img_model = ImageModel(cnn_output_dim=cnn_output_dim, pretrained=pretrained)
        self.tab_model = TabularModel(input_dim=tabular_input_dim,
                                      emb_dim=tabular_emb_dim,
                                      hidden_dim=256,
                                      dropout=dropout)
        fusion_dim = cnn_output_dim + tabular_emb_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, mask=None, tabular_data=None):
        img_feat = self.img_model(image)
        tab_feat = self.tab_model(tabular_data)
        fused = torch.cat([img_feat, tab_feat], dim=1)
        return self.classifier(fused)

class EarlyFusionRFC(nn.Module):
       def __init__(self, tabular_input_dim, num_classes, cnn_output_dim=512, pretrained=True, **kwargs):
           super().__init__()
           self.img_model = ImageModel(cnn_output_dim=cnn_output_dim, pretrained=pretrained)
           self.rfc_tabular = RandomForestTabular(input_dim=tabular_input_dim, num_classes=num_classes, **kwargs)
           fusion_dim = cnn_output_dim + num_classes
           self.classifier = nn.Sequential(
               nn.Linear(fusion_dim, 256),
               nn.ReLU(),
               nn.Linear(256, num_classes)
           )

       def forward(self, image, mask=None, tabular_data=None):
           img_feat = self.img_model(image)
           tab_feats_np = tabular_data.detach().cpu().numpy()
           tab_logits = torch.tensor(self.rfc_tabular.predict_proba(tab_feats_np), dtype=torch.float32, device=image.device)
           fused = torch.cat([img_feat, tab_logits], dim=1)
           return self.classifier(fused)
       

class LateFusionModel(nn.Module):
    """
    Late fusion classifier combining separate heads:
    Args:
        tabular_input_dim (int)
        num_classes (int)
        cnn_output_dim (int)
        tabular_emb_dim (int)
        dropout (float)
        pretrained (bool)
        fusion_method (str): 'concat' or 'weighted'
    """
    def __init__(self,
                 tabular_input_dim,
                 num_classes,
                 cnn_output_dim=512,
                 tabular_emb_dim=128,
                 dropout=0.5,
                 pretrained=True,
                 fusion_method='concat'):
        super(LateFusionModel, self).__init__()
        # image branch
        self.img_model = ImageModel(cnn_output_dim=cnn_output_dim, pretrained=pretrained)
        self.img_clf = nn.Sequential(
            nn.Linear(cnn_output_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        # tabular branch
        self.tab_model = TabularModel(input_dim=tabular_input_dim,
                                      emb_dim=tabular_emb_dim,
                                      hidden_dim=256,
                                      dropout=dropout)
        self.tab_clf = nn.Linear(tabular_emb_dim, num_classes)

        self.fusion_method = fusion_method
        if fusion_method == 'concat':
            self.fusion_clf = nn.Linear(num_classes * 2, num_classes)
        elif fusion_method == 'weighted':
            self.img_w = nn.Parameter(torch.tensor(0.5))
            self.tab_w = nn.Parameter(torch.tensor(0.5))
        else:
            raise ValueError("fusion_method must be 'concat' or 'weighted'")

    def forward(self, image, mask=None, tabular_data=None):
        img_feat = self.img_model(image)
        img_logits = self.img_clf(img_feat)

        tab_feat = self.tab_model(tabular_data)
        tab_logits = self.tab_clf(tab_feat)

        if self.fusion_method == 'concat':
            combined = torch.cat([img_logits, tab_logits], dim=1)
            return self.fusion_clf(combined)
        else:
            w_img = torch.sigmoid(self.img_w)
            w_tab = torch.sigmoid(self.tab_w)
            return w_img * img_logits + w_tab * tab_logits
        

class MaskImageModel(nn.Module):
    """
    ResNet18 backbone that optionally takes a mask, multiplies it with the image,
    and produces a feature vector.
    """
    def __init__(self, cnn_output_dim=512, pretrained=True):
        super(MaskImageModel, self).__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        self.cnn_output_dim = cnn_output_dim

    def forward(self, x_img, x_mask=None):
        
        if x_mask is not None:
            x_img = x_img * x_mask
        
        feat = self.backbone(x_img)
        return feat.view(feat.size(0), self.cnn_output_dim)

class MaskMultimodal(nn.Module):
    """
    Multimodal model incorporating image, mask, and tabular data.
    Image features are extracted with an optional mask application,
    tabular features with an MLP, and then concatenated for a final classifier.
    """
    def __init__(self,
                 tabular_input_dim,
                 num_classes,
                 cnn_output_dim=512,
                 tabular_emb_dim=128,
                 dropout=0.5,
                 pretrained=True):
        super(MaskMultimodal, self).__init__()

        self.frontview_image_model = ImageModel(cnn_output_dim=cnn_output_dim, pretrained=pretrained)
        self.image_model = MaskImageModel(cnn_output_dim=cnn_output_dim, pretrained=pretrained)
        self.tabular_model = TabularModel(input_dim=tabular_input_dim,
                                          emb_dim=tabular_emb_dim,
                                          hidden_dim=256,
                                          dropout=dropout)
        
        fusion_dim = cnn_output_dim + cnn_output_dim + tabular_emb_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, mask, tabular_data):
        frontview_img_feat = self.frontview_image_model(image)
        img_feat = self.image_model(image, mask)
        tab_feat = self.tabular_model(tabular_data)
        
        fused = torch.cat([frontview_img_feat, img_feat, tab_feat], dim=1)
        return self.classifier(fused)
    

class PretrainedBackboneWrapper(nn.Module):
    def __init__(self, pretrained_weights_path=None, freeze=True):
        super().__init__()
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.base_model.fc = nn.Identity()

        if pretrained_weights_path:
            print(f"Loading pretrained weights from: {pretrained_weights_path}")
            state_dict = torch.load(pretrained_weights_path, map_location='cpu')
            self.base_model.load_state_dict(state_dict, strict=False)

        if freeze:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.output_dim = 2048

    def forward(self, x):
        return self.base_model(x)

class MaskMultimodalWithPretrained(nn.Module):
    def __init__(self, tabular_input_dim, num_classes, 
                 tabular_emb_dim=128, dropout=0.5,
                 pretrained_backbone_path=None):
        super().__init__()

        self.frontview_image_model = PretrainedBackboneWrapper(
            pretrained_weights_path=pretrained_backbone_path,
            freeze=True
        )
        self.masked_image_model = PretrainedBackboneWrapper(
            pretrained_weights_path=pretrained_backbone_path,
            freeze=True
        )

        self.tabular_model = nn.Sequential(
            nn.Linear(tabular_input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, tabular_emb_dim),
            nn.ReLU()
        )

        fusion_dim = 2048 + 2048 + tabular_emb_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, mask, tabular_data):
        masked_image = image * mask
        frontview_features = self.frontview_image_model(image)
        masked_features = self.masked_image_model(masked_image)
        tabular_features = self.tabular_model(tabular_data)
        fused = torch.cat([frontview_features, masked_features, tabular_features], dim=1)
        return self.classifier(fused)
    

if __name__ == '__main__':
    load_dotenv()
    BASE_DIR = os.getenv('FILE_PATH')
    if not BASE_DIR:
        raise ValueError("FILE_PATH environment variable not set. Please create a .env file and set it.")

    DATA_PATH = os.path.join(BASE_DIR, 'Full_preprocessed_detailed_house.csv')
    print(f"Base Directory: {BASE_DIR}")
    print(f"Data CSV Path: {DATA_PATH}")

    numeric_cols = [ "procent_ingenomen", "area", "perimeter", "elongation",  "compactness", "huisnr_bag_letter"]
    categorical_cols = ['build_type']

    pipeline = MultimodalPipeline(
        model_class=MaskMultimodalWithPretrained,
        csv_path=DATA_PATH,
        image_base_dir=BASE_DIR,
        image_col='frontview_url',
        target_col='woningtype',
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        epochs=20, # You can specify the epochs here
        lr=1e-4,
        batch_size=32,
        useMask=True
    )

    pipeline.train() # or specify them here with 'epochs=X'
    pipeline.evaluate()
