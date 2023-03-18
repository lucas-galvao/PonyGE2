import re
import pandas as pd


def converter(row):

    new = []
    phenotype = row['phenotype']

    nconv, npool, nfc, nfcneuron = [int(i) for i in re.findall('\d+', phenotype.split('lr-')[0])]
    has_dropout = 'dropout' in phenotype
    has_batch_normalization = 'bnorm' in phenotype
    has_pool = 'pool' in phenotype
    learning_rate = phenotype.split('lr-')[1]

    for i in range(npool):

        for j in range(nconv):
            new.append('(Conv')
            if has_batch_normalization:
                new.append(' BNorm')
            new.append('),')

        if has_pool:
            new.append('(MaxPool')
            if has_dropout:
                new.append(' Dropout')
            new.append('),')

    new.append('(Flatten),')
    
    for i in range(nfc):
        new.append('(Fc %d' % nfcneuron)
        if has_dropout:
            new.append(' Dropout')
        new.append('),')

    new.append('(Lr %s)' % learning_rate)

    print(phenotype, ''.join(new))

    return ''.join(new)

csv = pd.read_csv('phenotypes.csv')
csv['phenotype'] = csv.apply(converter, axis=1)
csv.to_csv('output.csv', index=False)
print(csv)