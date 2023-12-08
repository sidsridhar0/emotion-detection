import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, zero_one_loss

#from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings('ignore')

# Fix the random seed for reproducibility
# !! Important !! : do not change this
seed = 1234
np.random.seed(seed)

emotion_dict = {'SURPRISE' : 1, 'DISGUST' : 2, 'HAPPINESS' : 3, 'FEAR' : 4, 'ANGER' : 5, 'CONTEMPT' : 6, 'NEUTRAL' : 7,  'SADNESS' : 8}

def load_images(directory):
    data_legend = './data/facial_expressions/data/legend.csv'
    df = pd.read_csv(data_legend, delimiter=',', names=['user.id','image','emotion'], header=1)
    labels = []
    images = []
    for img_name in os.listdir(directory):
        index = np.where(df.image == img_name)
        if len(index[0]) > 0:
            img_path = os.path.join(directory, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
            img = cv2.resize(img, (350, 350))  # Resize to a consistent size
            images.append(img)

            labels.append(emotion_dict[df.emotion[index[0][0]].upper()])

    return images, labels

def main():
    data_dir = './data/facial_expressions/images'
    print("loading images")
    X, y = np.array(load_images(data_dir))
    X = np.array([img.flatten() for img in X])
    y = y.astype('int')

    print('loaded images')

    rate = 0.01
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1, random_state=seed, shuffle=True)

    #accuracy = 0.5142439737034331
    #nn1 = MLPClassifier(random_state = seed, hidden_layer_sizes = [64, 64], activation = 'relu', 
    #            batch_size = 256, n_iter_no_change=100, max_iter=100, solver = 'sgd', learning_rate_init=rate).fit(X_tr, y_tr)
    
    #print('1:', nn1.score(X_te, y_te))

    #accuracy = 0.5142439737034331
    #nn2 = MLPClassifier(random_state = seed, hidden_layer_sizes = [350, 350], activation = 'relu', 
    #            batch_size = 256, n_iter_no_change=100, max_iter=100, solver = 'sgd', learning_rate_init=rate).fit(X_tr, y_tr)
    
    #print('2:', nn2.score(X_te, y_te))

    #accuracy = 0.5142439737034331
    nn3 = MLPClassifier(random_state = seed, hidden_layer_sizes = [64, 64], activation = 'relu', 
                batch_size = 128, n_iter_no_change=100, max_iter=100, solver = 'sgd', learning_rate_init=rate).fit(X_tr, y_tr)
    
    print('3:', nn3.score(X_te, y_te))

    #accuracy = 0.5142439737034331
    nn4 = MLPClassifier(random_state = seed, hidden_layer_sizes = [256, 256], activation = 'relu', 
                batch_size = 128, n_iter_no_change=100, max_iter=100, solver = 'sgd', learning_rate_init=rate).fit(X_tr, y_tr)
    
    print('4:', nn4.score(X_te, y_te))

    #accuracy = 
    nn5 = MLPClassifier(random_state = seed, hidden_layer_sizes = [350, 350], activation = 'relu', 
                batch_size = 256, n_iter_no_change=100, max_iter=100, solver = 'sgd', learning_rate_init=0.005).fit(X_tr, y_tr)
    
    print('5:', nn5.score(X_te, y_te))


if __name__ == '__main__':
    main()