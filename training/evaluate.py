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
    # Class 0: KEYWORD ("Hey Nova"), Class 1 & 2: Non-Keyword (Unknown & Background)
    y_binary = (y == LABEL_KEYWORD).astype(int)
    y_pred_binary = (y_pred == LABEL_KEYWORD).astype(int)
    
    cm_binary = confusion_matrix(y_binary, y_pred_binary)
    tn, fp, fn, tp = cm_binary.ravel()
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # True Positive Rate (Recall for Keyword)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0 # False Positive Rate (False Activation Rate)
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0 # False Negative Rate (Miss Rate)
    
    print("\n" + "="*50)
    print("              EdgeWake Evaluation Report          ")
    print("="*50)
    print(f"Overall Accuracy:  {accuracy*100:.2f}%")
    print(f"Weighted Precision: {precision*100:.2f}%")
    print(f"Weighted Recall:    {recall*100:.2f}%")
    print(f"Weighted F1-Score:  {f1*100:.2f}%")
    print("-"*50)
    print(f"Keyword Spotting Metrics:")
    print(f"  True Positive Rate (TPR):   {tpr*100:.2f}%")
    print(f"  False Positive Rate (FPR):  {fpr*100:.2f}%")
    print(f"  False Negative Rate (FNR):  {fnr*100:.2f}%")
    print("-"*50)
    print("Confusion Matrix:")
    print("Format:")
    print("               Predicted")
    print("               KW   Unk  BG")
    print(f"Actual KW:    {cm[0] if len(cm) > 0 else 'N/A'}")
    print(f"Actual Unk:   {cm[1] if len(cm) > 1 else 'N/A'}")
    print(f"Actual BG:    {cm[2] if len(cm) > 2 else 'N/A'}")
    print("-"*50)
    
    # Measure Latency on Mac
    print("Measuring inference latency...")
    warmup_runs = 10
    test_runs = 100
    
    # Warmup
    for _ in range(warmup_runs):
        _ = model.predict(X[:1], verbose=0)
        
    start_time = time.perf_counter()
    for _ in range(test_runs):
        _ = model.predict(X[:1], verbose=0)
    end_time = time.perf_counter()
    
    avg_latency_ms = (end_time - start_time) / test_runs * 1000
    print(f"Average Inference Latency on Mac: {avg_latency_ms:.2f} ms")
    print("="*50)
    
    # Return metrics for benchmarking
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tpr": tpr,
        "fpr": fpr,
        "fnr": fnr,
        "avg_latency_ms": avg_latency_ms,
        "params": model.count_params()
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate EdgeWake KWS Model")
    parser.add_argument("--model_path", type=str, default="models/edgewake_model.h5", help="Path to trained model file")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing evaluation WAV files")
    args = parser.parse_args()
    
    evaluate_model(model_path=args.model_path, data_dir=args.data_dir)
