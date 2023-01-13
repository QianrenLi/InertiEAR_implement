# ----------------------------
# Training Loop
# ----------------------------
import torch
from torch import nn
import torchvision.models as models
import numpy as np

def training(model, train_dl, val_dl, num_epochs):
    # Loss Function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0001,
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')

    # Repeat for each epoch
    for epoch in range(num_epochs):
        total_loss = 0.0
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Repeat for each batch in the training set
        for i, data in enumerate(train_dl):
            # Get the input features and target labels, and put them on the GPU
            inputs = data[0].to(device)
            labels = data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()
            total_loss += loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            if i % 20 == 0:  # print every 50 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0

        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = total_loss / num_batches
        acc = correct_prediction / total_prediction
        print(f'Epoch: {epoch}, Loss: {avg_loss:.5f}, Accuracy: {acc:.2f}')
        # Inference
        print(inference(model, val_dl, is_correlation=True))

    print('Finished Training')


# ----------------------------
# Inference
# ----------------------------

def inference(model, val_dl, is_correlation=True):
    model.eval()
    correct_prediction = 0
    total_prediction = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    correlation_matrix = np.zeros((10,10))

    # Disable gradient updates
    with torch.no_grad():
        for data in val_dl:
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Get predictions
            outputs = model(inputs)

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)
            
            if is_correlation:
                labels_np = labels.cpu().numpy()
                prediction_np = prediction.cpu().numpy()
                for i in range(len(labels_np)):
                    correlation_matrix[labels_np[i], prediction_np[i]] += 1

            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

    acc = correct_prediction / total_prediction
    print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')

    return correlation_matrix
