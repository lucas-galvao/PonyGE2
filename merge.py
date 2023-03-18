import sys, csv

arquivos = sys.argv[1:]

mapa = dict()

for arquivo in arquivos:
    print(arquivo)
    with open(arquivo, mode='r') as aqv:
        leitor = csv.DictReader(aqv)
        for linha in leitor:
            phenotype = linha['phenotype']
            accuracy = linha['accuracy']
            accuracy_sd = linha['accuracy_sd']
            f1_score = linha['f1_score']
            f1_score_sd = linha['f1_score_sd']
            if phenotype not in mapa or accuracy > mapa[phenotype][0]:
                mapa[phenotype] = (accuracy, accuracy_sd, f1_score, f1_score_sd)

with open('resultado.csv', mode='w+') as aqv:
    escritor = csv.DictWriter(aqv, fieldnames=['phenotype', 'accuracy', 'accuracy_sd', 'f1_score', 'f1_score_sd'])
    escritor.writeheader()
    for phenotype in mapa:
        accuracy, accuracy_sd, f1_score, f1_score_sd = mapa[phenotype]
        escritor.writerow({
            'phenotype': phenotype, 
            'accuracy': accuracy, 
            'accuracy_sd': accuracy_sd, 
            'f1_score': f1_score, 
            'f1_score_sd': f1_score_sd
        })