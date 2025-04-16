 😊 Facial Expression Detection & Emoji Recommendation

This project detects human facial expressions from images using a custom-built Convolutional Neural Network (CNN) and recommends a matching emoji in real time.

 📌 Overview

Facial expressions are a vital part of human communication. Our model identifies expressions like happy, sad, angry, etc., and maps them to intuitive emojis to enhance visual feedback in applications such as chat apps, virtual avatars, and accessibility tools.

 🚀 Features

- Real-time facial expression detection
- Emoji recommendation based on predicted emotion
- Custom CNN architecture (no transfer learning)
- Built using grayscale facial images for efficient processing
- Simple UI/CLI-based output (optional extension to GUI)

 🧠 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- MediaPipe – for face landmark detection
- NumPy, Matplotlib
- Raf-DB – facial expression dataset

 🏗️ Model Architecture

- 2 Convolutional Layers
- 2 MaxPooling Layers
- Flatten Layer
- Dense Layers with ReLU and Softmax activation
- Dropout for regularization

> Output Layer → 7 Classes: Happy, Sad, Angry, Surprise, Disgust, Fear, Neutral

 📊 Dataset

- RAF-DB: Real-world Affective Faces Database
- Grayscale images used for training
- Input image size: 64x64
- Dataset split: Training / Validation / Testing

 📈 Training Info

- Epochs: 30  
- Batch size: 32  
- Accuracy: ~94% training, ~76% validation  
- Loss function: Categorical Crossentropy  
- Optimizer: Adam  

 🔍 Testing Methodology

- Unit testing of model prediction using sample images
- Accuracy, loss tracking using validation dataset
- Epoch-wise performance observation to avoid overfitting

 🎭 Emoji Recommendation

Once an expression is detected, the model suggests a corresponding emoji:
| Expression | Emoji |
|------------|-------|
| Happy 😄   | 😄     |
| Sad 😢     | 😢     |
| Angry 😠   | 😠     |
| Surprise 😲| 😲     |
| Fear 😨    | 😨     |
| Disgust 🤢 | 🤢     |
| Neutral 😐 | 😐     |

 📷 Sample Output

> `Detected Expression: Happy`  
> `Recommended Emoji: 😄`


