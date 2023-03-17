import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
from random import randint
import tqdm
import pandas as pd


#device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def score(predictions, targets):
    score = []
    for i in range(len(predictions)):
        label_pred, label_target = np.argwhere(predictions[i]>0), np.argwhere(targets[i]>0)
        a = [int(i) for i in label_pred]
        b = [int(i) for i in label_target]
        score.append((len(set(a) & set(b)))/len(b))
        if i == 1:
            continue
            print(a,b,(len(set(a) & set(b)))/len(b))
    score = np.mean(np.array(score))
    return score

def compute_accuracy(model, dataloader, criterion):
    #criterion = nn.BCELoss()
    training_before = model.training
    model.eval()
    all_pred_torch = []
    all_predictions = []
    all_targets = []
    all_images = []
    for i_batch, batch in enumerate(dataloader):
        images, targets = batch['image'], batch['labels']
        targets = targets.type(torch.FloatTensor)
        image_batch = list(torch.unbind(images))
        all_images += image_batch
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        with torch.no_grad():
            predictions = model(images)
        all_pred_torch.append(criterion(predictions,targets))
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    predictions_numpy = np.concatenate(all_predictions, axis=0)
    targets_numpy = np.concatenate(all_targets, axis=0)
    for i in range(len(predictions_numpy)):
        Z = predictions_numpy[i]
        som=int(np.sum(targets_numpy[i]))
        Z2 = np.zeros(25)
        Z2[Z.argsort()[-som:]] = 1
        #mean = np.mean(predictions_numpy[i])
        #std = np.std(predictions_numpy[i])
        #Z = (predictions_numpy[i]-mean)/std
        predictions_numpy[i] = Z2
    test_loss = float(torch.mean(torch.tensor(all_pred_torch)))
    accuracy = score(predictions_numpy,targets_numpy)

    accuracy_BCE = nn.BCELoss()

    if training_before:
        model.train()

    return accuracy, predictions_numpy, targets_numpy, all_images, test_loss

def plot_results(predictions, targets,all_images, nombre_images):

    # Les noms des différents genre
    labels_name = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama',
                   'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'N/A', 'News', 'Reality-TV',
                   'Romance', 'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']

    # Plot des posters randoms du testdataset
    position_images_random = [randint(0, len(all_images)) for _ in range(0, nombre_images)]

    # les tracer
    for i in position_images_random :
        im = all_images[i]
        index_pred = np.reshape(np.argwhere(predictions[i]==1),-1)
        index_target = np.reshape(np.argwhere(targets[i] == 1), -1)
        title_pred = ' & '.join([labels_name[x] for x in index_pred])
        title_target = ' & '.join([labels_name[x] for x in index_target])
        plt.title('Prédiction : {} \n Vérité : {} '.format(title_pred, title_target))
        plt.imshow(im.permute(1, 2, 0))
        plt.show()

def custom_loss(predictions, targets ):
    for i in range(len(predictions)):
        mean = torch.mean(predictions[i])
        std = torch.std(predictions[i])
        Z = (predictions[i]-mean)/std
        Z[Z>=0.5] = 1
        Z[Z<0.5] = 0
        predictions[i] = Z
    score = []
    for i in range(len(predictions)):
        label_pred, label_target = torch.nonzero(predictions[i]), torch.nonzero(targets[i])
        a = [int(i) for i in label_pred]
        b = [int(i) for i in label_target]
        intersection = set(a) & set(b) 
        score.append(max((len(intersection)-(0.1)*(len(label_pred)-len(intersection))-0.3*(len(label_target)-len(intersection)))/len(label_target),0))
        if i == 1:
            
            print('lenin', len(intersection))
            print('fauxpos', (0.1)*(len(label_pred)-len(intersection)))
            print('fauxneg', 0.3*(len(label_target)-len(intersection)))
            print(label_pred,label_target,'score ==>',max((len(intersection)-(0.01)*(len(label_pred)-len(intersection))-0.1*(len(label_target)-len(intersection)))/len(label_target),0))
    score = torch.mean(torch.tensor(score,dtype=torch.float, requires_grad=True))
    return 1-score
    

def custom_loss2(a, b):
    loss = 0
    for i in range(len(a)):
        log = (-0.26/(a[i]+0.133)**2.5) + (0.26/(1+0.133)**2.5)
        loss += (torch.dot(b[i].float(),log)+0.75*(torch.dot((1-b[i]).float(),torch.log(1-a[i]))))/len(a[i])
    return -loss/len(a)


def plot_hist(test_dataloader_1):
    labels_name = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Docu.', 'Drama',
                   'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'N/A', 'News', 'Reality-TV',
                   'Romance', 'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']
    target = np.zeros(25)

    for i_batch, batch in enumerate(tqdm.tqdm(test_dataloader_1)):
        # Setup des images et de leurs targets
        for i in batch['labels']:
            targets = i
            a = targets.numpy()
            target += a
    dictionary = dict(zip(labels_name, target))
    sorted_dict = dict(sorted(dictionary.items(), key=lambda item: item[1], reverse=True))
    labels_name, target = zip(*sorted_dict.items())
    freq_series = pd.Series(target)
    my_colors = [(x / 26.0, x / 40.0, 0.75) for x in range(25)]
    plt.figure(figsize=(12, 8))
    ax = freq_series.plot(kind="bar", color=my_colors)
    ax.set_title("Fréquence des genres de films")
    ax.set_xlabel("Genres")
    ax.set_ylabel("Fréquence")
    ax.set_xticklabels(labels_name)
    rects = ax.patches

    # Make some labels.
    labels = [ '%.0f' % elem for elem in target]

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, height + 0.1, label, ha="center", va="bottom"
        )

    plt.show()

def accuracy_random(whole_dataset):
    '''
    -0.1
    :param whole_dataset:
    :return:
    '''
    targets_numpy = whole_dataset.labels
    predictions_numpy = np.zeros(25)
    argwhere = [randint(0, 24),randint(0, 24),randint(0, 24)]
    predictions_numpy[argwhere] = 1
    print(predictions_numpy)
    for i in range(targets_numpy.shape[0]-1):
        a = np.zeros(25)
        argwhere = [randint(0, 24), randint(0, 24), randint(0, 24)]
        a[argwhere] = 1
        predictions_numpy = np.vstack((a,predictions_numpy))
    for i in tqdm.tqdm(range(len(predictions_numpy))):
        mean = np.mean(predictions_numpy[i])
        std = np.std(predictions_numpy[i])
        Z = (predictions_numpy[i]-mean)/std
        Z[Z>=0.8] = 1
        Z[Z<0.8] = 0
        predictions_numpy[i] = Z
    accuracy = score(predictions_numpy,targets_numpy)
    return accuracy
