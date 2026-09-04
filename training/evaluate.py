import os
import argparse
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from training.dataset import load_dataset, LABEL_KEYWORD, LABEL_UNKNOWN, LABEL_BACKGROUND

def evaluate_model(model_path="models/edgewake_model.h5", data_dir="data"):
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found. Please train the model first.")
        return
        
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print("Loading evaluation dataset...")
    X, y = load_dataset(data_dir)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    print("Running predictions...")
    y_pred_probs = model.predict(X)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate Metrics
    accuracy = accuracy_score(y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    
    # Specific rates for Keyword vs Non-Keyword
    y_binary = (y == LABEL_KEYWORD).astype(int)
    y_pred_binary = (y_pred == LABEL_KEYWORD).astype(int)
    
    cm_binary = confusion_matrix(y_binary, y_pred_binary)
    tn, fp, fn, tp = cm_binary.ravel()
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    
    print("\n" + "="*50)
    print("              EdgeWake Evaluation Report          ")
    print("="*50)
    print(f"Overall Accuracy:  {accuracy*100:.2f}%")
    print(f"Weighted Precision: {precision*100:.2f}%")
    print(f"Weighted Recall:    {recall*100:.2f}%")
    print(f"Weighted F1-Score:  {f1*100:.2f}%")
    print("-"*50)
    print(f"Keyword Spotting Metrics:")
    print(f"  Keyword Sensitivity (TPR):  {tpr*100:.2f}%")
    print(f"  False Activation Rate (FPR): {fpr*100:.2f}%")
    print(f"  Keyword Miss Rate (FNR):     {fnr*100:.2f}%")
    print("-"*50)
    print("Per-Class Report:")
    print(classification_report(y, y_pred, target_names=['Keyword', 'Unknown Speech', 'Background Noise']))
    print("-"*50)
    print("Confusion Matrix:")
    print("               Predicted")
    print("               KW   Unk  BG")
    print(f"Actual KW:    {cm[0] if len(cm) > 0 else 'N/A'}")
    print(f"Actual Unk:   {cm[1] if len(cm) > 1 else 'N/A'}")
    print(f"Actual BG:    {cm[2] if len(cm) > 2 else 'N/A'}")
    print("-"*50)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tpr": tpr,
        "fpr": fpr,
        "fnr": fnr
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate EdgeWake KWS Model")
    parser.add_argument("--model_path", type=str, default="models/edgewake_model.h5", help="Path to trained model file")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing evaluation WAV files")
    args = parser.parse_args()
    
    evaluate_model(model_path=args.model_path, data_dir=args.data_dir)

