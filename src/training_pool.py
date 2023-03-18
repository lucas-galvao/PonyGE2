from stats.stats import stats, get_stats
from algorithm.parameters import params, set_params
from tensorflow.keras import datasets, layers, models, callbacks, optimizers
from tensorflow.keras import backend as K
from tensorflow.python.framework.errors_impl import ResourceExhaustedError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from datetime import datetime
import numpy as np
import tensorflow as tf
import re, os, sys, requests, time, random


def save_new_phenotypes(individuals):
    print('Saving new phenotypes.')
    data = {
        'grammar': params['GRAMMAR_NAME'],
        'dataset': params['DATASET_NAME'],
        'phenotypes': [ind.phenotype for ind in individuals]
    }
    try:
        r = requests.post('%s/phenotypes/save_new_phenotypes/' % params['METRICS_URL'], data=data)
    except BaseException as ex:
        print('save_new_phenotypes error:', ex)



def find_non_trained_phenotypes_and_mark_as_training():
    print('\tFinding non trained phenotypes')
    data = {
        'grammar': params['GRAMMAR_NAME'],
        'dataset': params['DATASET_NAME'],
    }
    try:
        r = requests.post('%s/phenotypes/find_non_trained/' % params['METRICS_URL'], data=data)
        result = r.json()
        return result['phenotypes']
    except BaseException as ex:
        print('find_non_trained_phenotypes error:', ex)
        return []
        


def find_trained_phenotype(phenotype):
    data = {
        'grammar': params['GRAMMAR_NAME'],
        'dataset': params['DATASET_NAME'],
        'phenotype': phenotype,
    }
    try:
        r = requests.post('%s/phenotypes/find_trained/' % params['METRICS_URL'], data=data)
        result = r.json()
        return result['metrics']
    except BaseException as ex:
        print('find_trained_phenotype error:', ex)
        return None


def mark_as_trained(phenotype, metrics):
    if metrics is None:
        metrics = {
            'accuracy': 0.0,
            'accuracy_sd': 0.0,
            'f1_score': 0.0,
            'f1_score_sd': 0.0,
            'time': 0.0,
            'time_sd': 0.0,
        }
    metrics['grammar'] = params['GRAMMAR_NAME']
    metrics['dataset'] = params['DATASET_NAME']
    metrics['phenotype'] = phenotype
    try:
        r = requests.post('%s/phenotypes/mark_as_trained/' % params['METRICS_URL'], data=metrics)
    except BaseException as ex:
        print('mark_as_trained error:', ex)


def load_data():

    dataset_name = params['DATASET_NAME']
    shape = params['DATASET_SHAPE']
    
    if dataset_name != 'cifar100':
        dataset = np.load('../../%s.npz' % dataset_name, allow_pickle=True)

    if dataset_name == 'cifar100':

        print('cifar100')

        train, test = datasets.cifar100.load_data()
        x_train, y_train = train
        x_test, y_test = test

        train_images = x_train.reshape((x_train.shape[0], *shape))
        train_images = train_images.astype("float") / 255.0
        train_labels = y_train

        test_images = x_test.reshape((x_test.shape[0], *shape))
        test_images = test_images.astype("float") / 255.0

        validation_images, test_images, validation_labels, test_labels = train_test_split(test_images, y_test, test_size=0.2, random_state=42)
    
    elif dataset_name == 'eurosat':
        
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

    if dataset_name != 'cifar100':
        dataset.close()

    return train_images, train_labels, validation_images, validation_labels, test_images, test_labels


def build_assuncao_model(phenotype):

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


def build_delima_model(phenotype):

    dataset_shape = params['DATASET_SHAPE']
    dataset_classes = params['DATASET_CLASSES']

    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=dataset_shape))

    parts = phenotype.split(',')
    dense_count = 0

    for part in parts:

        part = part.strip()

        if 'Conv2D' in part:

            _, filters, k_size, activation = part.split(' ')
            filters = int(filters)
            k_size = int(k_size)
            model.add(layers.Conv2D(filters, (k_size, k_size), activation=activation))

        elif 'MaxPooling2D' in part:

            _, p_size, padding = part.split(' ')
            p_size = int(p_size)
            model.add(layers.MaxPooling2D(pool_size=(p_size, p_size), padding=padding))

        elif 'AveragePooling2D' in part:

            _, p_size, padding = part.split(' ')
            p_size = int(p_size)
            model.add(layers.AveragePooling2D(pool_size=(p_size, p_size), padding=padding))

        elif 'Dropout' in part:

            _, rate = part.split(' ')
            rate = float(rate)
            model.add(layers.Dropout(rate))

        elif 'Dense' in part:

            _, neurons = part.split(' ')
            neurons = int(neurons)

            if dense_count == 0:
                model.add(layers.Flatten())
            
            model.add(layers.Dense(neurons))

            dense_count += 1

    model.add(layers.Dense(dataset_classes, activation='softmax'))

    opt = optimizers.Adam(learning_rate=0.01)

    return model, opt


def build_diniz_model(phenotype):

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


def build_cec21_model(phenotype):

    dataset_shape = params['DATASET_SHAPE']
    dataset_classes = params['DATASET_CLASSES']

    nconv, npool, nfc, nfcneuron = [int(i) for i in re.findall('\d+', phenotype.split('lr-')[0])]
    has_dropout = 'dropout' in phenotype
    has_batch_normalization = 'bnorm' in phenotype
    has_pool = 'pool' in phenotype
    learning_rate = float(phenotype.split('lr-')[1])

    # number of filters
    filter_size = 32

    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=dataset_shape))

    # Pooling
    for i in range(npool):

        # Convolutions
        for j in range(nconv):

            model.add(layers.Conv2D(filter_size, (3, 3), activation='relu', padding='same'))

            # Duplicate number of filters for each two convolutions
            if (((i + j) % 2) == 1): filter_size = filter_size * 2

            # Add batch normalization
            if has_batch_normalization:
                model.add(layers.BatchNormalization())

        # Add pooling
        if has_pool:
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
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

    model.add(layers.Dense(dataset_classes, activation='softmax'))

    opt = optimizers.Adam(learning_rate=learning_rate)

    return model, opt


def build_proposal_model(phenotype):

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


def build_model(phenotype, tpu_strategy):

    with tpu_strategy.scope():

        model, opt = None, None

        grammar_name = params['GRAMMAR_NAME']

        if grammar_name == 'proposal':
            model, opt = build_proposal_model(phenotype)
        elif grammar_name == 'diniz':
            model, opt = build_diniz_model(phenotype)
        elif grammar_name == 'delima':
            model, opt = build_delima_model(phenotype)
        elif grammar_name == 'assuncao':
            model, opt = build_assuncao_model(phenotype)
        elif grammar_name == 'cec21':
            model, opt = build_cec21_model(phenotype)

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


def train(model):

    if model is None:
        return None

    batch_size = 128

    accuracies, f1_scores, times = [], [], []

    train_images, train_labels, \
        validation_images, validation_labels, \
            test_images, test_labels = load_data()

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

    return {
        'accuracy': np.mean(accuracies),
        'accuracy_sd': np.std(accuracies),
        'f1_score': np.mean(f1_scores),
        'f1_score_sd': np.std(f1_scores),
        'time': np.round(np.mean(times)),
        'time_sd': np.round(np.std(times)),
    }


def go():
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
    while True:
        print('Checking if there are phenotypes to be trained...')
        phenotypes = find_non_trained_phenotypes_and_mark_as_training()
        for phenotype in phenotypes:
            print('\tFound:', phenotype)
            try:
                model = build_model(phenotype, tpu_strategy)
            except BaseException as e:
                model = None
                print(e)
            try:
                metrics = train(model)
            except (ValueError, ResourceExhaustedError) as e:
                print(e)
                metrics = None
            mark_as_trained(phenotype, metrics)
            print(metrics)
        time.sleep(15)


if __name__ == '__main__':
    set_params(sys.argv[1:])
    go()
