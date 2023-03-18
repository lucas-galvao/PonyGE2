from tensorflow.keras import datasets, layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import re


def evaluate(phenotype):
        
    print('FENOTIPO: %s' % phenotype)

    nconv, npool, nfc, nfcneuron = [int(i) for i in re.findall('\d+', phenotype.split('lr-')[0])]
    has_dropout = 'dropout' in phenotype
    has_batch_normalization = 'bnorm' in phenotype
    has_pool = 'pool' in phenotype
    learning_rate = float(phenotype.split('lr-')[1])

    # Carregando dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalizando
    train_images = train_images.astype("float") / 255.0
    test_images = test_images.astype("float") / 255.0

    lb = LabelBinarizer()
    train_labels = lb.fit_transform(train_labels)
    test_labels = lb.transform(test_labels)

    # num de filtros
    filter_size = 32

    # Iniciando o modelo da RN
    model = models.Sequential()

    try:
        
        # Pooling
        for i in range(npool):
    
            # Convolucoes
            for j in range(nconv):
    
                model.add(layers.Conv2D(filter_size, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))

                # Duplicate number of filters for each two convolutions
                if (((i + j) % 2) == 1): filter_size = filter_size * 2

                if has_batch_normalization:
                    model.add(layers.BatchNormalization())

            # Adiciona o pooling somente se estiver no fenotipo.
            if has_pool:
                model.add(layers.MaxPooling2D(pool_size=(2, 2)))
                if has_dropout:
                    model.add(layers.Dropout(0.25))

        model.add(layers.Flatten())

        # fully connected
        for i in range(nfc):
            model.add(layers.Dense(nfcneuron))
            model.add(layers.Activation('relu'))

        if has_dropout:
            model.add(layers.Dropout(0.5))

        model.add(layers.Dense(10, activation='softmax'))
        model.summary()

    except Exception as ex:
        print(ex)
        return 0

    opt = optimizers.Adam(lr=learning_rate)
    
    def f1_score(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1_val

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', f1_score])
    
    # history = model.fit(train_images, train_labels, epochs=70, batch_size=128, 
    #     validation_data=(test_images, test_labels), verbose=1)

    datagen = ImageDataGenerator(zoom_range=0.2, horizontal_flip=True)

    batch_size = 128
    
    history = model.fit(datagen.flow(train_images, train_labels, batch_size=batch_size),
            steps_per_epoch=train_images.shape[0]//batch_size, 
            epochs=400, 
            validation_data=(test_images, test_labels), 
            verbose=1)

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
        
    _, acuracia, f1 = model.evaluate(test_images, test_labels, verbose=2)
    
    print(acuracia, f1)


evaluate('(((conv*2)pool)*3)fc*1*256*lr-0.01')