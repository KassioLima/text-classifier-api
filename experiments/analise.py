import pandas as pd
from api.service import AiModelProduto, AiModelService, AiModelAssunto, AiModelProdutoAssunto
import matplotlib.pyplot as plt
import seaborn as sns
import asyncio

def obterMaxCaracteresFromModel() -> int:
    safetyValue = 40
    tokens = AiModelProduto.tokenizer.encode_plus(AiModelService.promptGigante, max_length=AiModelProduto.getModelAttr('tokens'), truncation=True, return_tensors="pt")
    return len(AiModelProduto.tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)) - safetyValue

def sumarizeModel(indexModel: int):
    df = pd.read_json("datasets/database-email-sep.json")
    
    AiModelProduto.init(modelIndex=indexModel)
    AiModelService.init(testing=True)
    
    maxLen = obterMaxCaracteresFromModel()
    
    df['length'] = df['DetalhesDaDemanda'].apply(len)
    
    onLimit = len(df[df['length'] <= maxLen])
    outLimit = len(df[df['length'] > maxLen])
    total = onLimit + outLimit
    
    return {
        "modelo": AiModelProduto.getModelAttr('name'),
        "tamanhoMaximoPrompt": maxLen,
        "promptsTotais": total,
        "promptsNoLimite": onLimit,
        "promptsForaDoLimite": outLimit,
        "percentualPromptsNoLimite": str(round((onLimit / total) * 100, 2)) + "%",
        "percentualPromptsForaDoLimite": str(round((outLimit / total) * 100, 2)) + "%"
    }

def compareModelsByTokensLimit():
    modelOptions = AiModelProduto.modelOptions
    modelOptions.sort(key=lambda x: x["tokens"])
    sumariesModel = [sumarizeModel(i) for i in range(len(modelOptions))]
    
    for sumary in sumariesModel:
        valor_total = sumary['promptsTotais']
        classe1 = sumary['promptsNoLimite']
        classe2 = sumary['promptsForaDoLimite']
        
        labels = ['Demandas com caracteres no limite (' + str(sumary['promptsNoLimite']) + ')',
                  'Demandas com caracteres fora do limite (' + str(sumary['promptsForaDoLimite']) + ')']
        sizes = [classe1, classe2]
        
        plt.figure(figsize=(10, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral'])
        plt.title("IA: " + sumary['modelo'] + " | Limite de caracteres: " + str(sumary['tamanhoMaximoPrompt']))
        
        total_text = f'Demandas: {valor_total}'
        plt.text(0, 0, total_text, ha='center', va='center', fontsize=12, color='black', fontweight='bold')
        
        plt.show()

def count_characters(text):
    return len(str(text))

def sumarizeDataset():
    df = pd.read_json("datasets/database-email-sep.json")
    df['text_length'] = df['DetalhesDaDemanda'].apply(count_characters)
    
    classSize = 1000
    
    # Definir intervalos de classes (por exemplo, intervalos de classSize caracteres)
    intervalos = range(0, df['text_length'].max() + classSize, classSize)
    
    # Criar coluna 'classe' com base nos intervalos
    df['classe'] = pd.cut(df['text_length'], bins=intervalos, right=False)
    
    # Contar observações por classe
    contagem_por_classe = df['classe'].value_counts().sort_index()
    
    # Filtrar classes com menos de 10 observações
    classes_validas = contagem_por_classe[contagem_por_classe >= 15].index
    
    # Filtrar DataFrame para incluir apenas observações nas classes válidas
    df_filtrado = df[df['classe'].isin(classes_validas)]
    
    # Remover classes com menos de 10 observações dos intervalos
    intervalos_filtrados = df_filtrado['classe'].unique()
    intervalos_filtrados = sorted(intervalos_filtrados, key=lambda x: x.left)
    
    # Configurar o estilo seaborn
    sns.set(style="whitegrid")
    
    # Criar um gráfico de barras usando seaborn
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=df_filtrado['classe'], color='blue', order=intervalos_filtrados)
    
    # Adicionar rótulos e título
    # plt.xlabel('Comprimento do Texto (Classes de 1000 caracteres)')
    plt.ylabel('Contagem de Demandas')
    plt.title('Distribuição do Comprimento do Texto por Classes de ' + str(classSize) + ' caracteres')
    
    for p, label in zip(ax.patches, contagem_por_classe):
        ax.annotate(label, (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10),
                    textcoords='offset points')
        
    # Exibir o gráfico de barras
    plt.xticks(rotation=0, ha='center')  # Rotacionar os rótulos do eixo x para melhor legibilidade
    
    plt.show()
    
def contarPercentualLinhas(rowNumber, dfLength):
    if (rowNumber / dfLength * 100) % 1 == 0:
        print(str(rowNumber).rjust(len(str(dfLength)), "0") + " / " + str(dfLength) + "\t" + str(int(rowNumber / dfLength * 100)) + "%")
    
async def obterPercentualAcertosModelo(modelo, df, targetColumn):
    dfAnalise = pd.DataFrame(columns=['milliseconds', 'textLen', 'score', 'isCorrect'])
    
    print("\nAnalizando acertos para \"" + targetColumn + "\"...")
    
    for index, row in df.iterrows():
        
        result = await AiModelService.classifyByAiModel(row['DetalhesDaDemanda'], modelo)
        
        if ',' not in targetColumn:
            nova_linha = [result['milliseconds'], result['processedCharacteres'], result['score'], (result['label'] == row[targetColumn])]
        else:
            columns = targetColumn.split(',')
            label = '{\'' + columns[0] + '\': \'' + row[columns[0]] + '\', \'' + columns[1] + '\': \'' + row[columns[1]] + '\'}'
            
            nova_linha = [result['milliseconds'], result['processedCharacteres'], result['score'], (result['label'] == label)]
        
        dfAnalise.loc[len(dfAnalise)] = nova_linha
        
    print("\n==========================RESULTADOS==========================\n")
    print(dfAnalise.describe())
    print("\nAcertos: " + str(round(dfAnalise['isCorrect'].sum() / len(dfAnalise) * 100, 2)) + "% (" + str(dfAnalise['isCorrect'].sum()) + " / " + str(len(dfAnalise)) + ")")
    
async def obterPercentualAcertosModelosCombinados(modelo1, modelo2, df, targetColumnModelo1, targetColumnModelo2):
    dfAnalise = pd.DataFrame(columns=['milliseconds', 'textLen', 'score_' + targetColumnModelo1, 'score_' + targetColumnModelo2, 'isCorrect'])
    
    print("\nAnalizando acertos para \"" + targetColumnModelo1 + "\" e \"" + targetColumnModelo2 + "\"...")
    
    for index, row in df.iterrows():
        
        resultModelo1 = (await AiModelService.classifyByAiModel(row['DetalhesDaDemanda'], modelo1))
        resultModelo2 = (await AiModelService.classifyByAiModel(row['DetalhesDaDemanda'], modelo2))
        
        nova_linha = [
            resultModelo1['milliseconds'] + resultModelo2['milliseconds'],
            resultModelo1['processedCharacteres'],
            resultModelo1['score'],
            resultModelo2['score'],
            (resultModelo1['label'] == row[targetColumnModelo1] and resultModelo2['label'] == row[targetColumnModelo2])]
        
        dfAnalise.loc[len(dfAnalise)] = nova_linha
    
    print("\n==========================RESULTADOS==========================\n")
    print(dfAnalise.describe())
    print("\nAcertos: " + str(round(dfAnalise['isCorrect'].sum() / len(dfAnalise) * 100, 2)) + "% (" + str(dfAnalise['isCorrect'].sum()) + " / " + str(len(dfAnalise)) + ")")
    
async def analizeModelsPerformance():
    AiModelProdutoAssunto.init(modelIndex=0)
    AiModelProduto.init(modelIndex=0)
    AiModelAssunto.init(modelIndex=0)
    AiModelService.init(testing=False)
    
    df = pd.read_json("datasets/database-email-sep.json")
    
    # df = df.sample(10)
    
    await obterPercentualAcertosModelo(AiModelProduto, df, 'produto')
    print("\n==============================================================")
    await obterPercentualAcertosModelo(AiModelAssunto, df, 'assunto')
    print("\n==============================================================")
    await obterPercentualAcertosModelosCombinados(AiModelProduto, AiModelAssunto, df, 'produto', 'assunto')
    print("\n==============================================================")
    await obterPercentualAcertosModelo(AiModelProdutoAssunto, df, 'produto,assunto')

sumarizeDataset()
compareModelsByTokensLimit()
asyncio.run(analizeModelsPerformance())
