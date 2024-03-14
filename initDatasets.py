import csv
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split


jsonDatasetName = "datasets/database-email-sep.json"  # dados extraídos do banco em 11/03/2024

datasetTipoName = "datasets/dataset-tipo.csv"
datasetProdutoName = "datasets/dataset-produto.csv"
datasetAssuntoName = "datasets/dataset-assunto.csv"

percentual_validacao = 0.2
sampleSize = 100


def removeLastLineBlank(datasetName: str):
    # remove a linha em branco no final do arquivo
    # with open(datasetName, 'rb+') as file:
    #     file.seek(-2, os.SEEK_END)
    #     file.truncate()
    #     file.close()
    pass


def preprocess_text(text: str):
    return json.dumps(text.strip())[1:-1].replace('"', '\\"')


def montarDataSetTrain(jsonSource, sourceColumnsToPreprocess, textColumn, targetColumn, datasetName, teste=False):
    global sampleSize

    df = pd.read_json(jsonSource)

    if teste:
        df = df.sample(sampleSize)

    if sourceColumnsToPreprocess is not None:
        for coluna_texto in sourceColumnsToPreprocess:
            df[coluna_texto] = df[coluna_texto].apply(preprocess_text)
    
    df = df[[textColumn, targetColumn]]
    df.columns = ['text', 'target']
    
    df = df.drop_duplicates()
            
    df.to_csv(datasetName, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')
    
    removeLastLineBlank(datasetName)

    print("\nDataset csv gerado com sucesso! (" + datasetName + ")\n")


def montarDataSetEvaluation(datasetName, evaluationDatasetName):
    global percentual_validacao
    column_class = "target"

    # Carregue o conjunto de dados de treino
    df_treino = pd.read_csv(datasetName, encoding='utf-8')

    contagem_classes = df_treino[column_class].value_counts()
    classes_com_apenas_um_membro = contagem_classes[contagem_classes == 1].index.tolist()

    # remove do dataframe de treino os elementos que pertencem à classes com apenas 1 membro
    df_treino = df_treino[~df_treino[column_class].isin(classes_com_apenas_um_membro)]

    # Use train_test_split para dividir o conjunto de treino em treino e validação
    df_treino, df_validacao = train_test_split(df_treino, test_size=percentual_validacao, stratify=df_treino[column_class])

    # Garanta que o conjunto de validação tenha o mesmo número de classes que o conjunto de treino
    classes_validacao = df_validacao[column_class].unique()
    df_treino = df_treino[df_treino[column_class].isin(classes_validacao)]

    verificarNumeroDeClasses(df_treino, df_validacao, column_class)

    # Salve os conjuntos de treino e validação em arquivos CSV separados
    df_treino.to_csv(datasetName, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')
    df_validacao.to_csv(evaluationDatasetName, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')
    
    removeLastLineBlank(datasetName)
    removeLastLineBlank(evaluationDatasetName)

    print("Datasets separados com sucesso!")


def montarDataSetTrainEvaluation(jsonDataset, sourceColumnsToPreprocess, textColumn, targetColumn, datasetTrainName, datasetValidationName, teste=False):
    if teste:
        datasetTrainName = parseToTesteName(datasetTrainName)
        datasetValidationName = parseToTesteName(datasetValidationName)
    
    montarDataSetTrain(jsonSource=jsonDataset, sourceColumnsToPreprocess=sourceColumnsToPreprocess, textColumn=textColumn, targetColumn=targetColumn, datasetName=datasetTrainName, teste=teste)
    verificarConsistenciaDataSet(datasetName=datasetTrainName)
    
    montarDataSetEvaluation(datasetName=datasetTrainName, evaluationDatasetName=datasetValidationName)
    verificarConsistenciaDataSet(datasetName=datasetValidationName)


def verificarConsistenciaDataSet(datasetName):
    dataFrame = pd.read_csv(datasetName, encoding='utf-8')

    for index, row in dataFrame.iterrows():
        if len(row) != 2:
            raise Exception(f"Erro na linha {index}: Número incorreto de colunas - {len(row)} colunas encontradas | {datasetName}")

        for column, value in row.items():
            if type(value) != type(""):
                raise Exception(f"Tipo na linha {index}, coluna {column}: {type(value)} | {datasetName}")

    if dataFrame.isnull().any().any():
        raise Exception(f"O DataFrame contém valores nulos | {datasetName}")

    if dataFrame.duplicated().any():
        raise Exception(f"O DataFrame contém linhas duplicadas | {datasetName}")


def addSuffix(name: str, suffix: str):
    return name.split(".")[0] + "-" + suffix + "." + name.split(".")[1]


def parseToTesteName(name: str):
    return addSuffix(name.split("/")[0] + "/teste/" + name.split("/")[1], 'teste')


def parseToTrainName(name: str):
    return addSuffix(name, 'train')


def parseToEvaluationName(name: str):
    return addSuffix(name, 'evaluation')


def verificarNumeroDeClasses(dataframe1, dataframe2, column_class):

    num_classes_dataframe1 = len(dataframe1[column_class].unique())
    num_classes_dataframe2 = len(dataframe2[column_class].unique())

    str_num_classes_dataframe1 = "Classes dataframe_1: " + str(num_classes_dataframe1)
    str_num_classes_dataframe2 = "Classes dataframe_2:  " + str(num_classes_dataframe2)

    if num_classes_dataframe2 == num_classes_dataframe1:
        print(str_num_classes_dataframe1)
        print(str_num_classes_dataframe2)
    else:
        raise Exception(f"A quantidade de classes está ERRADA.\nA quantidade de classes em ambos os dataframes devem ser IGUAIS!\n{str_num_classes_dataframe1}\n{str_num_classes_dataframe2}")


def datasetsConfigs():
    global jsonDatasetName
    global datasetTipoName
    global datasetProdutoName
    global datasetAssuntoName
    
    return [
        {
            "jsonDatasetName": jsonDatasetName,
            "sourceColumnsToPreprocess": ['DetalhesDaDemanda'],
            "textColumn": 'DetalhesDaDemanda',
            "targetColumn": 'TipoDeDemanda',
            "datasetName": datasetTipoName,
        },
        {
            "jsonDatasetName": jsonDatasetName,
            "sourceColumnsToPreprocess": ['DetalhesDaDemanda'],
            "textColumn": 'DetalhesDaDemanda',
            "targetColumn": 'produto',
            "datasetName": datasetProdutoName,
        },
        {
            "jsonDatasetName": jsonDatasetName,
            "sourceColumnsToPreprocess": ['DetalhesDaDemanda'],
            "textColumn": 'DetalhesDaDemanda',
            "targetColumn": 'assunto',
            "datasetName": datasetAssuntoName,
        }
    ]


def montarDatasets():
    for datasetConfig in datasetsConfigs():
        montarDataSetTrainEvaluation(jsonDataset=datasetConfig['jsonDatasetName'], sourceColumnsToPreprocess=datasetConfig['sourceColumnsToPreprocess'], textColumn=datasetConfig['textColumn'], targetColumn=datasetConfig['targetColumn'], datasetTrainName=addSuffix(datasetConfig['datasetName'], 'train'), datasetValidationName=addSuffix(datasetConfig['datasetName'], 'evaluation'))
        montarDataSetTrainEvaluation(jsonDataset=datasetConfig['jsonDatasetName'], sourceColumnsToPreprocess=datasetConfig['sourceColumnsToPreprocess'], textColumn=datasetConfig['textColumn'], targetColumn=datasetConfig['targetColumn'], datasetTrainName=addSuffix(datasetConfig['datasetName'], 'train'), datasetValidationName=addSuffix(datasetConfig['datasetName'], 'evaluation'), teste=True)


if __name__ == '__main__':
    montarDatasets()
