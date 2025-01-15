# Image Classification with Keras

## **Project Overview**
This project utilizes a pre-trained Keras model to classify images into predefined categories, such as "Mountain" or "Sea," using deep learning techniques.

---

## **Project Structure**
```
project/
|
├── keras_model.h5          # Pre-trained model
├── labels.txt              # Classification labels
├── mt_rainier.jpg          # Example image for testing
├── main.py                 # Core Python script
├── README.md               # Project description
```

---

## **Technologies Used**
- TensorFlow/Keras for the model.
- Pillow for image processing.
- NumPy for data preparation.

---

## **How to Run**
1. Clone the repository:
   ```bash
   git clone https://github.com/username/project.git
   cd project
   ```

2. Install dependencies:
   ```bash
   pip install tensorflow pillow numpy
   ```

3. Run the script:
   ```bash
   python main.py
   ```

4. The output will display:
   - **Class Name**: Predicted label.
   - **Confidence Score**: Probability of correctness.

---

## **Example Output**
```
Class: Mountain
Confidence Score: 0.95
```

---

## **Future Improvements**
- Train the model on a larger, more diverse dataset.
- Add a GUI for ease of use.
- Support additional categories.

---



