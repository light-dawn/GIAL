# encoding=utf-8
from network.resnet import ResNet18

if __name__ == "__main__":
    model = ResNet18(num_classes=20)
    weight_dict = model.state_dict()
    target_layers = ["layer1.1.conv2.weight", "layer2.1.conv2.weight",
                     "layer3.1.conv2.weight", "layer4.1.conv2.weight"]
    target_weights = [weight_dict[layer] for layer in target_layers]
    for weight in target_weights:
        print(weight.shape)


