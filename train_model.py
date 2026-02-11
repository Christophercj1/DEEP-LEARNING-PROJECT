import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader 
import os
import mlflow 
import mlflow.pytorch


mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("cardetectionmodel")

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform=transforms.Compose(
    [
        transforms.Resize((224,224)),transforms.ToTensor()
    ]
)


Basedir=os.path.dirname(os.path.abspath(__file__))
datatrain=os.path.join(Basedir,"data","train")
datatest=os.path.join(Basedir,"data","test")


train_data=datasets.ImageFolder(datatrain,transform=transform)
test_data=datasets.ImageFolder(datatest,transform=transform)

batchsize=32
train_loader=DataLoader(train_data,batch_size=batchsize)
test_loader=DataLoader(test_data,batch_size=batchsize)


model=models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad=False

model.fc=nn.Linear(model.fc.in_features,2)
model.to(device)

learningrate=0.01
loss_function=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.fc.parameters(), lr=learningrate)

with mlflow.start_run():
    epochs=3
    mlflow.log_param("model","resnet18")
    mlflow.log_param("epochs",epochs)
    mlflow.log_param("batchsize",batchsize)
    mlflow.log_param("learningrate",learningrate)



   
    for epoch in range(epochs):
        model.train()
        loss=0
        correct=0
        total=0
        for images,labels in train_loader:
            images,labels=images.to(device),labels.to(device)   
            optimizer.zero_grad()
            output=model(images)
            loss_value=loss_function(output,labels)
            loss_value.backward()
            optimizer.step()  #importance level update(weights)
            loss+=loss_value.item()
            _,prediction=torch.max(output,1)
            correct+=(prediction==labels).sum().item()
            total+=labels.size(0)
        accuracy=correct/total
        print(f"epoch {epoch+1}loss:{loss_value.item():.4f}")
        mlflow.log_metric("trainloss",loss)
        mlflow.log_metric("trainaccuracy",accuracy)
        mlflow.pytorch.log_model(model,"carmodel")
        print("accuracy",accuracy)

torch.save(model.state_dict(),"carmodel.pth")     #store the model


