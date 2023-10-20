import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from model import CNN  # Make sure to import your CNN model

# Define the transformations for the input image
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load the trained model
model = CNN()
model.load_state_dict(torch.load('your_model.pth'))
model.eval()

# Load the MNIST test dataset
test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)

# Choose an image from the test dataset (you can change the index)
test_image, test_label = test_dataset[0]  # Change the index as needed

# test_image_np = test_image.squeeze(0).numpy().reshape(784, 1)

test_image_np = test_image.squeeze(0).numpy().reshape(1, 784, order='F').transpose()


# Make a prediction
with torch.no_grad():
    output = model(test_image.unsqueeze(0))  # Add a batch dimension

# Convert the output tensor to a NumPy array
output_array = output.numpy()

# Get the predicted label
predicted_label = np.argmax(output_array)

print(f'Actual digit: {test_label}')
print(f'Predicted digit: {predicted_label}')

# Save test_image_np as a .txt file
np.savetxt('./para_BN_W&b_txt/test_image.hpp', test_image_np, fmt='%.6f,', delimiter=',')



