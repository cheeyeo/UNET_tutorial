import os
import logging
import tqdm
import torch
from torch.optim.lr_scheduler import OneCycleLR
import segmentation_models_pytorch as smp
import numpy as np
import matplotlib.pyplot as plt
from segmentation.model import UNet
from segmentation.data import get_dataloaders, get_datasets
from segmentation.utils import meanIoU
from segmentation.utils import train_validate_model
from segmentation.utils import evaluate_model
from segmentation.utils import visualize_predictions
from segmentation.utils import train_id_to_color


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%d-%m-%Y %H:%M',
                        filename='trainer.log',
                        filemode='w')
                        
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format without the date time for console 
    formatter = logging.Formatter('%(name)-12s %(levelname)-4s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    logger1 = logging.getLogger('UNet Main')
    logger2 = logging.getLogger('UNet Trainer')

    # Load dataset
    logger1.info("Loading dataset...")
    images = np.load("segmentation/dataset/image_180_320.npy")
    labels = np.load("segmentation/dataset/label_180_320.npy")
    logger1.info(f"Images shape {images.shape}")
    logger1.info(f"Labels shape {labels.shape}")

    train_set, val_set, test_set = get_datasets(images, labels)
    sample_image, sample_label = train_set[0]
    logger1.info(f"{len(train_set)} train images, {len(val_set)} validation images, {len(test_set)} test images")
    logger1.info(f"Input shape={sample_image.shape}, Label shape={sample_label.shape}")

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(train_set, val_set, test_set)

    criterion = smp.losses.DiceLoss("multiclass", classes=[0, 1, 2], log_loss=True, smooth=1.0)

    # MODEL HYPERPARAMETERS
    N_EPOCHS = 5
    NUM_CLASSES = 3
    MAX_LR = 3e-4
    MODEL_NAME = "UNet baseline"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=3, out_channels=3, layer_channels=[64, 128, 256, 512]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=MAX_LR)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        epochs=N_EPOCHS,
        steps_per_epoch=len(train_dataloader),
        pct_start=0.3,
        div_factor=10,
        anneal_strategy="cos"
    )

    # Train loop
    output_path = os.path.join(os.getcwd(), "artifacts")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    logger1.info("Train model ...")
    results = train_validate_model(model, N_EPOCHS, MODEL_NAME, criterion, optimizer, device, train_dataloader, val_dataloader, meanIoU, 'meanIoU', NUM_CLASSES, lr_scheduler=scheduler, output_path=output_path, savefig=True, logger=logger2)

    logger1.info("Evaluating model on test set...")
    model.load_state_dict(torch.load(f"{output_path}/{MODEL_NAME}.pt", map_location=device))

    _, test_metric = evaluate_model(model, test_dataloader, criterion, meanIoU, NUM_CLASSES, device)
    logger1.info(f"Model has {test_metric} mean IoU in test set")

    num_test_samples = 2
    _, axes = plt.subplots(num_test_samples, 3, figsize=(3 * 6, num_test_samples*4))

    visualize_predictions(model, test_set, axes, device, numTestSamples=num_test_samples, id_to_color=train_id_to_color, savefig=True)