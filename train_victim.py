import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data_path = '/Users/drkrm/Documents/IRIS/obd_emissions_data.csv'
onnx_path = '/Users/drkrm/Documents/IRIS/victim_model.onnx'

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Could not find the dataset at {data_path}. Run generate_data.py first!")

df = pd.read_csv(data_path)
X = df.drop(columns=['Emission_Anomaly']).values
y = df['Emission_Anomaly'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test).unsqueeze(1)
 
class VictimModel(nn.Module):
    def __init__(self):
        super(VictimModel, self).__init__()
        
        self.fc1 = nn.Linear(5, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model = VictimModel()
criterion = nn.BCELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.01)


print("Training the baseline Unoptimized model")
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")


model.eval()
with torch.no_grad():
    predictions = model(X_test_t)
    predicted_classes = (predictions > 0.5).float()
    accuracy = (predicted_classes == y_test_t).float().mean().item()
    print(f"\n Baseline Model Accuracy: {accuracy * 100:.2f}%")


dummy_input = torch.randn(1, 5) 
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_path, 
    export_params=True,
    opset_version=11,
    input_names=['sensor_input'],
    output_names=['anomaly_output']
)
print(f"\n Unoptimized model exported successfully to:\n {onnx_path}")
