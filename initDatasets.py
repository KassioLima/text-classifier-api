import csv
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split


jsonDatasetTrainName = "datasets/database.json"  # dados extraídos do banco em 07/03/2024
datasetTrainName = "datasets/dataset-train.csv"
datasetEvaluationName = "datasets/dataset-evaluation.csv"
percentual_validacao = 0.2
sampleSize = 5000


def removeLastLineBlank(datasetName: str):
    # remove a linha em branco no final do arquivo
    with open(datasetName, 'rb+') as file:
        file.seek(-2, os.SEEK_END)
        file.truncate()
        file.close()


def preprocess_text(text):
    return json.dumps(text)[1:-1].replace('"', '\\"')


def montarDataSetTrain(jsonSource, sourceColumnsToPreprocess, datasetName, teste=False):
    global sampleSize

    df = pd.read_json(jsonSource)

    if teste:
        df = df.sample(sampleSize)

    if sourceColumnsToPreprocess is not None:
        for coluna_texto in sourceColumnsToPreprocess:
            df[coluna_texto] = df[coluna_texto].apply(preprocess_text)

    df.to_csv(datasetName, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')
    
    removeLastLineBlank(datasetName)

    print("\nDataset csv gerado com sucesso! (" + datasetName + ")\n")


def montarDataSetEvaluation(datasetName, evaluationDatasetName, column_class):
    global percentual_validacao

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


def montarDatasets():
    global jsonDatasetTrainName
    global datasetTrainName
    global datasetEvaluationName

    montarDataSetTrain(jsonDatasetTrainName, ['text'], datasetTrainName, False)
    verificarConsistenciaDataSet(datasetTrainName)
    montarDataSetEvaluation(datasetTrainName, datasetEvaluationName, "target")
    verificarConsistenciaDataSet(datasetEvaluationName)

    montarDataSetTrain(jsonDatasetTrainName, ['text'], parseToTesteName(datasetTrainName), True)
    verificarConsistenciaDataSet(parseToTesteName(datasetTrainName))
    montarDataSetEvaluation(parseToTesteName(datasetTrainName), parseToTesteName(datasetEvaluationName), "target")
    verificarConsistenciaDataSet(parseToTesteName(datasetEvaluationName))


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


def parseToTesteName(name: str):
    return name.split("/")[0] + "/teste/" + name.split("/")[1].split(".")[0] + "-teste." + name.split(".")[1]


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


if __name__ == '__main__':
    montarDatasets()
