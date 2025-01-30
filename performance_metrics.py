import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
from sklearn.metrics import (
    auc,
    confusion_matrix,  
    classification_report,  
    roc_curve
    )

# load json and create model
def load_model(json_path, weights_path):

    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_path)
    return loaded_model

def test_vs_pred(test_path, categories, model):
    original_labels = []
    predicted_labels = []
    predicted_score = []
    original_score = []
    for category in categories:
        path_to_category = os.path.join(test_path, category)
        for sub in tqdm(os.listdir(path_to_category)):
            path_to_sub = os.path.join(path_to_category, sub)
            for img_path in tqdm(os.listdir(path_to_sub)):
                path_to_image = os.path.join(path_to_sub, img_path)
                img = image.load_img(path_to_image, target_size=(160,160))
                img = image.img_to_array(img)
                img = np.expand_dims(img,axis=0)
                img = img / 255.0
                original_labels.append(category)
                prediction = model.predict(img, verbose=0)
                predicted_score.append(prediction[0][0])
                if prediction < 0.5:
                    predicted_labels.append('real')
                    original_score.append(0)
                else:
                    predicted_labels.append('spoof')
                    original_score.append(1)
    df = pd.DataFrame({"test": original_labels, "pred": predicted_labels,
                       "test_score": original_score, "pred_score": predicted_score})
    return df

def plot_auc_roc_curve(fpr, tpr, roc_auc, img_name):
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(img_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add the --json_path argument
    parser.add_argument("--json_path", "-jp", type=str, required=True,
                        help="The path to the model config architecutes stored in a JSON file.")
    parser.add_argument("--weights_path", "-wp", type=str, required=True,
                        help="The path to the trained model weights.")
    parser.add_argument("--img_name", "-img", type=str, required=True,
                        help="The name of image file which contains auc-roc curve.")
    
    args = parser.parse_args()
    json_path = args.json_path
    weights_path = args.weights_path
    img_name = args.img_name
    # json_path = 'research_antispoofing_model_mobilenet.json'
    # weights_path = "model_weights/finalyearproject_antispoofing_model_194-0.962132.h5"
    categories = ['real','spoof']
    test_path = os.path.join(os.getcwd(), "final_antispoofing/test")
    loaded_model = load_model(json_path=json_path, 
                              weights_path = weights_path)
    test_vs_pred_df = test_vs_pred(test_path = test_path, 
                                   categories = categories, 
                                   model = loaded_model)
    print(confusion_matrix(test_vs_pred_df['test'], test_vs_pred_df['pred']))
    print(classification_report(test_vs_pred_df['test'], test_vs_pred_df['pred']))
    fpr, tpr, thresholds = roc_curve(test_vs_pred_df['test_score'], test_vs_pred_df['pred_score'])
    roc_auc = auc(fpr, tpr)
    plot_auc_roc_curve(fpr, tpr, roc_auc, img_name)




