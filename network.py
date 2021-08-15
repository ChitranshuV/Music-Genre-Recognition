import torch.nn as nn
import torchvision.models as models


class CNN:

    # .requires_grad=True, if we are training from scratch or finetuning
    # Otherwise while using pretrained model, we turn .requires_grad=False
    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    # Initializing pretrained Inception V3 model
    def initialize_model(
        self, model_name, num_classes, feature_extract, use_pretrained=True
    ):
        if model_name == "inception":
            """Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model_ft = models.inception_v3(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 299
        else:
            print("This model isn't here yet, exiting...")
            exit()

        return model_ft, input_size
