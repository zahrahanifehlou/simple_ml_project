from model import load_model
import numpy as np

def main():
    model = load_model()

    # Example: one flower
    sample = np.array([[3,4.7,3.2,1.3,0.2]])

    prediction = model.predict(sample)
    print(f"Predicted class: {prediction[0]}")

if __name__ == "__main__":
    main()