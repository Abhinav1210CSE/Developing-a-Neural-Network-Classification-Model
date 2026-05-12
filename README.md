
# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
Include the neural network model diagram.
<img width="1058" height="855" alt="590580087-57bde504-8629-468f-a2d0-6e6a0b5fb196" src="https://github.com/user-attachments/assets/9111e711-bbfe-4b86-a720-509b35572cb3" />


## DESIGN STEPS
STEP 1:
Load the dataset, drop unnecessary columns (like ID), handle missing values, and apply Label Encoding to categorical features and the target (Segmentation).

STEP 2:
Split the data into training and testing sets, then normalize features using StandardScaler.

STEP 3:
Convert the processed data into PyTorch tensors and create DataLoaders for batch processing.

STEP 4:
Build a feedforward neural network with fully connected layers and ReLU activation, ending with a multi-class output layer.

STEP 5:
Train the model using CrossEntropyLoss and Adam optimizer with forward pass, loss calculation, backpropagation, and updates.

STEP 6:
Evaluate using accuracy, confusion matrix, and classification report, and test predictions on sample input.

## PROGRAM

### Name: Abhinav CS

### Register Number:212224040005


```
# Define Neural Network(Model1)
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        #Include your code here
        self.fc1=nn.Linear(input_size,32)
        self.fc2=nn.Linear(32,16)
        self.fc3=nn.Linear(16,8)
        self.fc4=nn.Linear(8,4)


    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        return x

# Training Loop
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoc in range(epochs):
      for input, labels in train_loader:
        optimizer.zero_grad()
        outputs=model(input)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

  #Include your code here

    if (epochs + 1) % 10 == 0:
        print(f'Epoch [{epochs+1}/{epochs}], Loss: {loss.item():.4f}')


# Initialize model
model = PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

```


### Dataset Information
Include screenshot of the dataset.
<img width="1754" height="207" alt="image" src="https://github.com/user-attachments/assets/61a68f9a-0cea-47f5-a16a-ea9ae8888bb3" />


### OUTPUT

## Confusion Matrix
<img width="421" height="177" alt="image" src="https://github.com/user-attachments/assets/ec76ae2d-2d0c-4f7d-b291-740bfb01b419" />

<img width="669" height="566" alt="image" src="https://github.com/user-attachments/assets/d6956ada-f2fc-4f45-aa04-599ef13546df" />

Include confusion matrix here

## Classification Report
Include classification report here
<img width="620" height="246" alt="image" src="https://github.com/user-attachments/assets/50f87baf-de93-4639-b048-fd9ab1b501b6" />


### New Sample Data Prediction
Include your sample input and output here
<img width="548" height="100" alt="image" src="https://github.com/user-attachments/assets/c56ab627-00cf-4cbf-aaea-f1b9132e94a1" />


## RESULT
Thus developing a neural network classification model for the given data set has been executed successfully.
