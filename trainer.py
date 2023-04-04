import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Load the TuSimple lane detection dataset
data_dir = "path/to/TuSimple/dataset"
X = []
y = []
for foldername in os.listdir(data_dir):
    folderpath = os.path.join(data_dir, foldername)
    if not os.path.isdir(folderpath):
        continue
        
    for filename in os.listdir(folderpath):
        if not filename.endswith('.jpg'):
            continue
            
        img = cv2.imread(os.path.join(folderpath, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X.append(img)
        
        label = filename[:-4] + '.lines.txt'
        with open(os.path.join(folderpath, label), 'r') as f:
            lines = f.readlines()
        lines = [list(map(float, line.strip().split())) for line in lines]
        y.append(lines)

# Convert the data to numpy arrays
X = np.array(X)
y = np.array(y)

# Preprocess the data
X = X / 255.0

# Split the data into training and testing sets
split_idx = int(0.8 * len(X))
X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]

# Define the neural network architecture
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(720, 1280, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(80, activation='linear'))

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='mse')

# Train the model on the lane detection dataset
history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model on the test set
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Make predictions on new data
predictions = model.predict(X_test)

# Visualize the predicted lane markings
fig, axs = plt.subplots(4, 4, figsize=(16, 16))
for i in range(16):
    img = X_test[i]
    ax = axs[i//4, i%4]
    ax.imshow(img)
    
    true_lines = y_test[i]
    for line in true_lines:
        x1, y1, x2, y2 = map(int, line)
        ax.plot([x1, x2], [y1, y2], color='green')
    
    pred_lines = predictions[i]
    for line in pred_lines:
        x1, y1, x2, y2 = map(int, line)
        ax.plot([x1, x2], [y1, y2], color='red')
    ax.axis('off')
    
model.save("./models")

plt.show()
