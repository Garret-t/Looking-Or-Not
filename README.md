# Looking-Or-Not

## Goal
Take input from the webcam and determine whether or not I am looking at the screen.

## Plan
- Gather data from google images of people looking forward and those that are looking to the side. 
- Split the data into train, test, and validation sets
- Create a convolutional neural network using PyTorch
- Try out different convnet architectures for transfer learning
- Augment the data to adjust for overfitting
- Train the data and adjust hyperparameters
- Implement the model into a program taking in my webcam input
- Profit?

## Files
All the data I collected from google images is in the datasets folder. The Jupyter Notebook file is used to generate the trained model, which I did through the use of Google Collab. That model is saved as a .pth file. Then the python file loads the .pth file and uses the model by taking in images via OpenCV.

## Results
I was able to achieve roughly 70% accuracy with the GoogleNet inception architecture. Bigger architectures such as the Resnet architecture proved better accuracy (about 80%) at the cost of extremely slow performance. However, with a more capable machine, the ResNet architecture would likely be better suited for the job.  

Video feed from the webcam is input into a convolutional neural network and then decerns whether or not the person is looking at the camera.

![](https://i.imgur.com/xwNRJ1H.gif)
