from torchvision.models import alexnet
from modeling.networks.resnet18 import feature_resnet18, feature_resnet50, feature_wide_resnet50, feature_resnet50_2

NET_OUT_DIM = {'alexnet': 256, 'resnet18': 512, 'resnet50': 2048, "wide_resnet50": 2048}


def build_feature_extractor(backbone):
    if backbone == "alexnet":
        print("Feature extractor: AlexNet")
        return alexnet(pretrained=True).features
    elif backbone == "resnet18":
        print("Feature extractor: ResNet-18")
        return feature_resnet18()
    elif backbone == "resnet50":
        print("Feature extractor: ResNet-50")
        return feature_resnet50()
    elif backbone == "wide_resnet50":
        print("Feature extractor: WideResNet-50")
        return feature_resnet50_2()
    else:
        raise NotImplementedError