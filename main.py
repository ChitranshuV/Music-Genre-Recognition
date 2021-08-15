from dataset import FMADataset
from network import CNN
from train import training
import torch
import torch.nn as nn
import torch.optim as optim


if __name__ == "__main__":
    ANNOTATIONS_FILE = "/run/media/high/Edu/Music Genre Classification Project/Music-Genre-Recognition/track-genre.csv"
    AUDIO_DIR = "/run/media/high/Edu/Music Genre Classification Project/fma_small/"
    sample_rate = 22050
    num_samples = sample_rate * 30
    batch_size = 16
    num_epochs = 25
    num_classes = 10
    lr = 0.001
    momentum = 0.9
    feature_extract = True
    model_name = "inception"

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    # Initialize the model for this run
    model_ft, input_size = CNN().initialize_model(
        model_name, num_classes, feature_extract, use_pretrained=True
    )

    # Print the model we just instantiated
    print(model_ft)
    # Send the model to GPU
    model_ft = model_ft.to(device)

    print("Initializing Datasets and Dataloaders...")

    fma_train = FMADataset(
        ANNOTATIONS_FILE,
        AUDIO_DIR,
        sample_rate,
        num_samples,
        input_size,
        device,
        "Train",
    )
    fma_val = FMADataset(
        ANNOTATIONS_FILE,
        AUDIO_DIR,
        sample_rate,
        num_samples,
        input_size,
        device,
        "Test",
    )

    fma_dataset = {"train": fma_train, "val": fma_val}

    dataloaders_dict = {
        x: torch.utils.data.DataLoader(
            fma_dataset[x], batch_size=batch_size, shuffle=True, num_workers=4
        )
        for x in ["train", "val"]
    }

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=lr, momentum=momentum)
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = training().train_model(
        model_ft,
        dataloaders_dict,
        criterion,
        optimizer_ft,
        num_epochs=num_epochs,
        is_inception=(model_name == "inception"),
    )

    training().accuracy_plot(hist, num_epochs)
