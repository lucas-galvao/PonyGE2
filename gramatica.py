import re, csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import backend as K 
from tensorflow.keras import datasets, layers, models, callbacks, optimizers

filename = '/pesquisa/output.csv'

def get_metrics(phenotype):
    accuracy, accuracy_sd, f1_score, f1_score_sd = None, None, None, None
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == phenotype:
                accuracy = float(row[1])
                accuracy_sd = float(row[2])
                f1_score = float(row[3])
                f1_score_sd = float(row[4])
                break
    return accuracy, accuracy_sd, f1_score, f1_score_sd


def save_metrics(phenotype, accuracy, accuracy_sd, f1_score, f1_score_sd):
    with open(filename, mode='a') as file:
        writer = csv.writer(file)
        writer.writerow([phenotype, accuracy, accuracy_sd, f1_score, f1_score_sd])


def load_dataset():

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    validation_images, test_images, validation_labels, test_labels = train_test_split(test_images, test_labels, test_size=0.2, random_state=42)
    
    train_images = train_images.astype("float") / 255.0
    test_images = test_images.astype("float") / 255.0
    validation_images = validation_images.astype("float") / 255.0

    lb = LabelBinarizer()
    train_labels = lb.fit_transform(train_labels)
    validation_labels = lb.transform(validation_labels)
    test_labels = lb.transform(test_labels)
    
    return train_images, train_labels, test_images, test_labels, validation_images, validation_labels


def build_model(phenotype):

    model = models.Sequential()

    filter_size = 32
    nconvs = 0
    optimizer = None

    model.add(layers.InputLayer(input_shape=(32, 32, 3)))

    for block in phenotype.split(','):
        
        if 'Conv' in block:

            if nconvs == 2:
                filter_size *= 2
                nconvs = 0

            model.add(layers.Conv2D(filter_size, (3, 3), activation='relu', padding='same'))

            if 'BNorm' in block:
                model.add(layers.BatchNormalization())

            nconvs += 1
        
        if 'MaxPool' in block:
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            
            if 'Dropout' in block:
                model.add(layers.Dropout(0.25))

        if 'Flatten' in block:
            model.add(layers.Flatten())

        if 'Fc' in block:
            args = re.findall('\d+', block)
            model.add(layers.Dense(int(args[0])))
            model.add(layers.Activation('relu'))

            if 'Dropout' in block:
                model.add(layers.Dropout(0.5))

        if 'Lr' in block:
            args = re.findall('\d+\.\d+', block)
            optimizer = optimizers.Adam(lr=float(args[0]))


    model.add(layers.Dense(10, activation='softmax'))

    def f1_score(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1_val

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', f1_score])
    model.summary()

    return model


def train_model(model):

    accuracies, f1_scores = [], []

    train_images, train_labels, test_images, \
        test_labels, validation_images, validation_labels = load_dataset()

    # Train three times
    for i in range(3):

        print('Trainning %s of 3' % (i + 1))

        # Early Stop when bad networks are identified        
        es = callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10, baseline=0.5)

        model.fit(train_images, train_labels, epochs=70, batch_size=128, 
            validation_data=(validation_images, validation_labels), callbacks=[es])
        
        _, accuracy, f1_score = model.evaluate(test_images, test_labels, verbose=1)

        accuracies.append(accuracy)
        f1_scores.append(f1_score)

        if i == 0 and accuracy < 0.5:
            break

    return np.mean(accuracies), np.std(accuracies), np.mean(f1_scores), np.std(f1_scores)


def evaluate(variables):

    phenotype = variables[0]

    accuracy, accuracy_sd, f1_score, f1_score_sd = get_metrics(phenotype)

    if accuracy is None and f1_score is None:

        print('Phenotype not yet trained. Building...')

        model = build_model(phenotype)

        if model:
            accuracy, accuracy_sd, f1_score, f1_score_sd = train_model(model)
        else:
            accuracy, accuracy_sd, f1_score, f1_score_sd = 0.0, 0.0, 0.0, 0.0

        save_metrics(phenotype, accuracy, accuracy_sd, f1_score, f1_score_sd)

    print(accuracy, accuracy_sd, f1_score, f1_score_sd)

    return accuracy, f1_score