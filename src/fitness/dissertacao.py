from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff
from stats.stats import stats
from tensorflow.keras import datasets, layers, models, callbacks, optimizers
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from datetime import datetime
import numpy as np
import re, csv, os, requests, time
import tensorflow as tf


class dissertacao(base_ff):

    maximise = True
    multi_objective = True

    def __init__(self):
        super().__init__()
        self.execution_id = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
        self.num_obj = 2
        fit = base_ff()
        fit.maximise = True
        self.fitness_functions = [fit, fit]
        self.default_fitness = [float('nan'), float('nan')]
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
        tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
        self.tpu_strategy = tpu_strategy

    def load_data(self):

        dataset_name = params['DATASET_NAME']
        shape = params['DATASET_SHAPE']
        dataset = np.load('../../%s.npz' % dataset_name, allow_pickle=True)

        if dataset_name == 'eurosat':
            
            print('eurosat')
            
            train = dataset['train'].tolist()

            train_images, train_labels = train['image'], train['label']

            train_images = train_images.reshape((train_images.shape[0], *shape))
            train_images = train_images.astype("float") / 255.0

            train_images, test_images, train_labels, test_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)
            validation_images, test_images, validation_labels, test_labels = train_test_split(test_images, test_labels, test_size=0.2, random_state=42)

        elif dataset_name in ['pathmnist', 'octmnist', 'organmnist_axial']:
            
            print('medmnist:', dataset_name)
            
            train_images = dataset['train_images']
            validation_images = dataset['val_images']
            test_images = dataset['test_images']
            train_labels = dataset['train_labels']
            validation_labels = dataset['val_labels']
            test_labels = dataset['test_labels']

            if shape[2] == 1:
                train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
                validation_images = validation_images.reshape((validation_images.shape[0], 28, 28, 1))
                test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

            train_images = train_images.astype("float") / 255.0
            test_images = test_images.astype("float") / 255.0
            validation_images = validation_images.astype("float") / 255.0

        else:
            
            print('outros:', dataset_name)
            
            train = dataset['train'].tolist()
            test = dataset['test'].tolist()

            train_images, test_images, train_labels, test_labels = train['image'], test['image'], train['label'], test['label']

            train_images = train_images.reshape((train_images.shape[0], *shape))
            train_images = train_images.astype("float") / 255.0

            test_images = test_images.reshape((test_images.shape[0], *shape))
            test_images = test_images.astype("float") / 255.0

            validation_images, test_images, validation_labels, test_labels = train_test_split(test_images, test_labels, test_size=0.2, random_state=42)

        lb = LabelBinarizer()
        train_labels = lb.fit_transform(train_labels)
        validation_labels = lb.transform(validation_labels)
        test_labels = lb.transform(test_labels)

        dataset.close()

        return train_images, train_labels, validation_images, validation_labels, test_images, test_labels


    def get_metrics(self, phenotype):

        accuracy, accuracy_sd, f1_score, f1_score_sd, time, time_sd = None, None, None, None, None, None

        r = requests.get('%s/api/metrics/' % params['METRICS_URL'], params={
            'grammar': params['GRAMMAR_NAME'],
            'dataset': params['DATASET_NAME'],
            'phenotype': phenotype,
        })
        data = r.json()

        if len(data):
            data = data[0]
            accuracy = float(data['accuracy'])
            accuracy_sd = float(data['accuracy_sd'])
            f1_score = float(data['f1_score'])
            f1_score_sd = float(data['f1_score_sd'])
            time = float(data['time'])
            time_sd = float(data['time_sd'])

        return accuracy, accuracy_sd, f1_score, f1_score_sd, time, time_sd

    def save_metrics(self, phenotype, accuracy, accuracy_sd, f1_score, f1_score_sd, time, time_sd):

        data = {
            'grammar': params['GRAMMAR_NAME'],
            'dataset': params['DATASET_NAME'],
            'phenotype': phenotype,
            'accuracy': accuracy,
            'accuracy_sd': accuracy_sd,
            'f1_score': f1_score,
            'f1_score_sd': f1_score_sd,
            'time': time,
            'time_sd': time_sd,
        }
        r = requests.post('%s/api/metrics/' % params['METRICS_URL'], json=data)

    def save_step(self, phenotype, accuracy, f1_score, time):

        data = {
            'execution': self.execution_id,
            'grammar': params['GRAMMAR_NAME'],
            'dataset': params['DATASET_NAME'],
            'generation': stats['gen'],
            'phenotype': phenotype,
            'accuracy': accuracy,
            'f1_score': f1_score,
            'time': time,
        }
        r = requests.post('%s/api/steps/' % params['METRICS_URL'], json=data)

    def build_assuncao_model(self, phenotype):

        dataset_shape = params['DATASET_SHAPE']
        dataset_classes = params['DATASET_CLASSES']

        model = models.Sequential()
        model.add(layers.InputLayer(input_shape=dataset_shape))

        learning_rate = None
        parts = phenotype.split(', ')

        for part in parts:

            sections = part.split(' ')

            if 'layer:conv' in part:
                padding = sections[4]
                activation = sections[5]
                num_filters, filter_shape, stride, bias, bnorm, merge = [int(i) for i in re.findall('\d+', part)]
                model.add(layers.Conv2D(num_filters, (filter_shape, filter_shape), strides=(stride, stride), use_bias=bool(bias), activation=activation, padding=padding))
                if bnorm:
                    model.add(layers.BatchNormalization())
            elif 'layer:pool' in part:
                padding = sections[3]
                kernel_size, stride = [int(i) for i in re.findall('\d+', part)]
                if 'avg' in part:
                    model.add(layers.AveragePooling2D(pool_size=(kernel_size, kernel_size), strides=(stride, stride), padding=padding))
                elif 'max' in part:
                    model.add(layers.MaxPooling2D(pool_size=(kernel_size, kernel_size), strides=(stride, stride), padding=padding))
            elif 'layer:fc' in part:
                if 'Flatten' not in ''.join([str(l.__class__) for l in model.layers]):
                    model.add(layers.Flatten())
                activation = sections[1]
                if 'softmax' in activation:
                    model.add(layers.Dense(dataset_classes, activation=activation))
                else:
                    num_units, bias = [int(i) for i in re.findall('\d+', part)]
                    model.add(layers.Dense(num_units, activation=activation, use_bias=bool(bias)))
            elif 'learning' in part:
                learning_rate = float(sections[1])

        opt = optimizers.SGD(learning_rate=learning_rate)

        return model, opt


    def build_delima_model(self, phenotype):

        dataset_shape = params['DATASET_SHAPE']
        dataset_classes = params['DATASET_CLASSES']

        model = models.Sequential()
        model.add(layers.InputLayer(input_shape=dataset_shape))

        parts = phenotype.split(',')
        num_dense = len(re.findall('Dense', phenotype))
        dense_count = 0

        for part in parts:

            part = part.strip()

            if 'Conv2D' in part:

                _, filters, k_size, activation = part.split(' ')
                filters = int(filters)
                k_size = int(k_size)
                model.add(layers.Conv2D(filters, (k_size, k_size), activation=activation))

            elif 'MaxPooling2D' in part or 'AveragePooling2D' in part:

                _, p_size, padding = part.split(' ')
                p_size = int(p_size)
                model.add(layers.MaxPooling2D(pool_size=(p_size, p_size), padding=padding))

            elif 'Dropout' in part:

                _, rate = part.split(' ')
                rate = float(rate)
                model.add(layers.Dropout(rate))

            elif 'Dense' in part:

                _, neurons = part.split(' ')
                neurons = int(neurons)

                if dense_count == 0:
                    model.add(layers.Flatten())

                if dense_count + 1 == num_dense:
                    model.add(layers.Dense(dataset_classes, activation='softmax'))
                else:
                    model.add(layers.Dense(neurons))

                dense_count += 1

        opt = optimizers.Adam(learning_rate=0.001)

        return model, opt


    def build_diniz_model(self, phenotype):

        dataset_shape = params['DATASET_SHAPE']
        dataset_classes = params['DATASET_CLASSES']

        model = models.Sequential()
        model.add(layers.InputLayer(input_shape=dataset_shape))

        nconv, npool, nfc = [int(i) for i in re.findall('\d+', phenotype)]
        has_pool = 'pool' in phenotype

        filter_size = 32

        # Pooling
        for i in range(npool):

            # Convolutions
            for j in range(nconv):

                model.add(layers.Conv2D(filter_size, (3, 3), activation='relu', padding='same'))

                # Duplicate number of filters for each two convolutions
                if (((i + j) % 2) == 1): filter_size = filter_size * 2

            # Add pooling
            if has_pool:
                model.add(layers.MaxPooling2D(pool_size=(2, 2)))


        model.add(layers.Flatten())

        # fully connected
        for i in range(nfc):
            model.add(layers.Dense(256))
            model.add(layers.Activation('relu'))

        model.add(layers.Dense(dataset_classes, activation='softmax'))

        opt = optimizers.Adam(learning_rate=0.01)

        return model, opt


    def build_proposal_model(self, phenotype):

        dataset_shape = params['DATASET_SHAPE']
        dataset_classes = params['DATASET_CLASSES']

        model = models.Sequential()
        model.add(layers.InputLayer(input_shape=dataset_shape))

        learning_rate = None
        filter_size = 32
        nconvs = 0

        nblocks = int(phenotype[0])

        for n in range(nblocks):

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


        for block in phenotype.split(','):

            if 'Flatten' in block:
                model.add(layers.Flatten())

            if 'Fc' in block:

                nfc, neurons = re.findall('\d+', block)

                for n in range(int(nfc)):
                    model.add(layers.Dense(int(neurons)))
                    model.add(layers.Activation('relu'))

                if 'Dropout' in block:
                    model.add(layers.Dropout(0.5))

            if 'Softmax' in block:
                model.add(layers.Dense(dataset_classes, activation='softmax'))

            if 'Lr' in block:
                args = re.findall('\d+\.\d+', block)
                learning_rate = float(args[0])


        opt = optimizers.Adam(learning_rate=learning_rate)

        return model, opt


    def build_model(self, phenotype):

        with self.tpu_strategy.scope():

            model, opt = None, None

            grammar_name = params['GRAMMAR_NAME']

            if grammar_name == 'proposal':
                model, opt = self.build_proposal_model(phenotype)
            elif grammar_name == 'diniz':
                model, opt = self.build_diniz_model(phenotype)
            elif grammar_name == 'delima':
                model, opt = self.build_delima_model(phenotype)
            elif grammar_name == 'assuncao':
                model, opt = self.build_assuncao_model(phenotype)

            model.summary()

            # F1 Score metric function
            def f1_score(y_true, y_pred):
                true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
                predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
                precision = true_positives / (predicted_positives + K.epsilon())
                recall = true_positives / (possible_positives + K.epsilon())
                f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
                return f1_val

            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', f1_score])

        return model


    def train_model(self, model):

        if model is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        batch_size = 128

        accuracies, f1_scores, times = [], [], []

        train_images, train_labels, \
            validation_images, validation_labels, \
                test_images, test_labels = self.load_data()

        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size, drop_remainder=True)
        validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels)).batch(batch_size, drop_remainder=True)

        # Train three times
        for i in range(3):

            # To free memory on google colab.
            if K.backend() == 'tensorflow':
                K.clear_session()

            print('Training %s of 3' % (i + 1))

            # Early Stop when bad networks are identified
            es = callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10, baseline=0.5)

            start_time = time.time()

            model.fit(train_ds,
                epochs=30,
                batch_size=batch_size,
                verbose=0,
                validation_data=validation_ds,
                callbacks=[es],
                )

            end_time = round(time.time() - start_time)

            loss, accuracy, f1_score = model.evaluate(test_images, test_labels, verbose=1)

            accuracies.append(accuracy)
            f1_scores.append(f1_score)
            times.append(end_time)

            if i == 0 and accuracy < 0.5:
                break

        return np.mean(accuracies), np.std(accuracies), \
                np.mean(f1_scores), np.std(f1_scores), \
                np.round(np.mean(times)), np.round(np.std(times))

    def evaluate(self, ind, **kwargs):

        print('PHENOTYPE: %s' % ind.phenotype)
        print('DATASET: %s' % params['DATASET_NAME'])
        print('GRAMMAR: %s' % params['GRAMMAR_NAME'])

        accuracy, accuracy_sd, f1_score, f1_score_sd, time, time_sd = self.get_metrics(ind.phenotype)

        if accuracy is None and f1_score is None:

            print('Phenotype not yet trained. Building...')

            with self.tpu_strategy.scope():
                try:
                    model = self.build_model(ind.phenotype)
                except:
                    model = None

            try:
                accuracy, accuracy_sd, f1_score, f1_score_sd, time, time_sd = self.train_model(model)
            except ValueError as e:
                print(e)
                accuracy, accuracy_sd, f1_score, f1_score_sd, time, time_sd = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            self.save_metrics(ind.phenotype, accuracy, accuracy_sd, f1_score, f1_score_sd, time, time_sd)

        print(accuracy, accuracy_sd, f1_score, f1_score_sd)

        self.save_step(ind.phenotype, accuracy, f1_score, time)

        return accuracy, f1_score

    @staticmethod
    def value(fitness_vector, objective_index):

        if not isinstance(fitness_vector, list):
            return float("inf")

        return fitness_vector[objective_index]
