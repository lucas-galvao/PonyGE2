from fitness.base_ff_classes.base_ff import base_ff
from utilities.fitness.get_video_data import get_video_frames
from tensorflow.keras import datasets, layers, models, callbacks, optimizers
from tensorflow.keras import backend as K 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import re, csv, os


class violence_detection(base_ff):

    maximise = True
    multi_objective = True

    def __init__(self):
        super().__init__()
        self.filename = os.path.join('pesquisa', 'phenotypes.csv') #'/pesquisa/phenotypes.csv'
        self.num_obj = 2
        fit = base_ff()
        fit.maximise = True
        self.fitness_functions = [fit, fit]
        self.default_fitness = [float('nan'), float('nan')]


    def load_data(self):
        # Setting variables
        IMAGE_SHAPE = (50, 50)
        N_FRAMES = 15
        train_dir = '../datasets/violence-detection-dataset/videos/train/cam1'
        test_dir = '../datasets/violence-detection-dataset/videos/test/cam1'

        # Load dataset
        train_images, train_labels = get_video_frames(train_dir, N_FRAMES, training = True, format_frame = True, output_size = IMAGE_SHAPE)
        test_images, test_labels = get_video_frames(test_dir, N_FRAMES, training = False, format_frame = True, output_size = IMAGE_SHAPE)
        validation_images, test_images, validation_labels, test_labels = train_test_split(test_images, test_labels, test_size=0.2, random_state=42)
        
        # Normalizing
        lb = LabelBinarizer()
        train_labels = lb.fit_transform(train_labels)
        validation_labels = lb.transform(validation_labels)
        test_labels = lb.transform(test_labels)
        
        return train_images, train_labels, test_images, test_labels, validation_images, validation_labels

    def get_metrics(self, phenotype):
        accuracy, accuracy_sd, f1_score, f1_score_sd = None, None, None, None
        with open(self.filename, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == phenotype:
                    accuracy = float(row[1])
                    accuracy_sd = float(row[2])
                    f1_score = float(row[3])
                    f1_score_sd = float(row[4])
                    break
        return accuracy, accuracy_sd, f1_score, f1_score_sd

    def save_metrics(self, phenotype, accuracy, accuracy_sd, f1_score, f1_score_sd):
        with open(self.filename, mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([phenotype, accuracy, accuracy_sd, f1_score, f1_score_sd])

    def build_model(self, phenotype):

        # To free memory on google colab.
        if K.backend() == 'tensorflow':
            K.clear_session()

        nconv, npool, nfc, nfcneuron = [int(i) for i in re.findall('\d+', phenotype.split('lr-')[0])]
        has_dropout = 'dropout' in phenotype
        has_batch_normalization = 'bnorm' in phenotype
        has_pool = 'pool' in phenotype
        learning_rate = float(phenotype.split('lr-')[1])

        # number of filters
        #filter_size = 32
        filter_size = 16

        model = models.Sequential()

        try:
        
            # Pooling
            for i in range(npool):
        
                # Convolutions
                for j in range(nconv):
        
                    #model.add(layers.Conv2D(filter_size, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
                    model.add(layers.Conv3D(filter_size, (3, 3, 3), activation='relu', padding='same', input_shape=(15, 50, 50, 3)))

                    # Duplicate number of filters for each two convolutions
                    if (((i + j) % 2) == 1): filter_size = filter_size * 2

                    # Add batch normalization
                    if has_batch_normalization:
                        model.add(layers.BatchNormalization())

                # Add pooling
                if has_pool:
                    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
                    # Add dropout
                    if has_dropout:
                        model.add(layers.Dropout(0.25))

            model.add(layers.Flatten())

            # fully connected
            for i in range(nfc):
                model.add(layers.Dense(nfcneuron))
                model.add(layers.Activation('relu'))

            if has_dropout:
                model.add(layers.Dropout(0.5))

            #model.add(layers.Dense(10, activation='softmax'))
            model.add(layers.Dense(1, activation='softmax'))
            # model.summary()

        except Exception as ex:
            # Some NN topologies are invalid
            print(ex)
            return None

        opt = optimizers.Adam(lr=learning_rate)

        # F1 Score metric function
        def f1_score(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            recall = true_positives / (possible_positives + K.epsilon())
            f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
            return f1_val

        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', f1_score])
        
        return model


    def train_model(self, model):

        accuracies, f1_scores = [], []

        train_images, train_labels, test_images, \
            test_labels, validation_images, validation_labels = self.load_data()

        # Train three times
        for i in range(3):

            print('Trainning %s of 3' % (i + 1))

            # Early Stop when bad networks are identified        
            es = callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10, baseline=0.5)

            model.fit(train_images, train_labels, epochs=30, batch_size=32, 
                validation_data=(validation_images, validation_labels), callbacks=[es])
            
            loss, accuracy, f1_score = model.evaluate(test_images, test_labels, verbose=1)

            accuracies.append(accuracy)
            f1_scores.append(f1_score)

            if i == 0 and accuracy < 0.5:
                break

        return np.mean(accuracies), np.std(accuracies), np.mean(f1_scores), np.std(f1_scores)

    def evaluate(self, ind, **kwargs):

        print('PHENOTYPE: %s' % ind.phenotype)

        # accuracy, accuracy_sd, f1_score, f1_score_sd = self.get_metrics(ind.phenotype)
        accuracy, f1_score = None, None
        
        if accuracy is None and f1_score is None:

            print('Phenotype not yet trained. Building...')

            model = self.build_model(ind.phenotype)

            if model:
                accuracy, accuracy_sd, f1_score, f1_score_sd = self.train_model(model)
            else:
                accuracy, accuracy_sd, f1_score, f1_score_sd = 0.0, 0.0, 0.0, 0.0

            self.save_metrics(ind.phenotype, accuracy, accuracy_sd, f1_score, f1_score_sd)

        print(accuracy, accuracy_sd, f1_score, f1_score_sd)

        return accuracy, f1_score

    @staticmethod
    def value(fitness_vector, objective_index):
        """
        This is a static method required by NSGA-II for sorting populations
        based on a given fitness function, or for returning a given index of a
        population based on a given fitness function.

        :param fitness_vector: A vector/list of fitnesses.
        :param objective_index: The index of the desired fitness.
        :return: The fitness at the objective index of the fitness vecror.
        """

        if not isinstance(fitness_vector, list):
            return float("inf")

        return fitness_vector[objective_index]
