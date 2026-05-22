import cobra
import pandas as pd
import warnings
import sys

# Ignora avisos comuns, como os de inconsistências em modelos SBML,
# para manter o foco na análise principal.
warnings.filterwarnings("ignore")

def load_metabolic_model(model_path):
    """
    Carrega o modelo metabólico e valida a sua estrutura.
    
    Args:
        model_path (str): Caminho para o arquivo do modelo (e.g., 'iJO1366.xml').
        
    Returns:
        cobra.Model: O objeto do modelo COBRA.
    """
    try:
        model = cobra.io.read_sbml_model(model_path)
        print(f"Modelo '{model.id}' carregado com sucesso.")
        return model
    except FileNotFoundError:
        print(f"Erro: O arquivo do modelo em '{model_path}' não foi encontrado.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}", file=sys.stderr)
        return None

def load_shadow_price_data(excel_path):
    """
    Carrega os dados de preços sombra de um arquivo Excel,
    garantindo que todas as abas sejam lidas.

    Args:
        excel_path (str): Caminho para o arquivo Excel de entrada.

    Returns:
        dict: Um dicionário onde as chaves são os nomes das abas
              e os valores são DataFrames.
    """
    try:
        excel_sheets = pd.read_excel(excel_path, sheet_name=None)
        print(f"Dados de preços sombra carregados de '{excel_path}'.")
        return excel_sheets
    except FileNotFoundError:
        print(f"Erro: O arquivo de dados em '{excel_path}' não foi encontrado.", file=sys.stderr)
        print("Certifique-se de que o primeiro script foi executado e gerou o arquivo de saída.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Erro ao ler o arquivo Excel: {e}", file=sys.stderr)
        return None

def calculate_reaction_shadow_price(reaction, sp_map):
    """
    Calcula o preço sombra médio para uma reação com base nos preços
    sombra de seus metabólitos.

    Args:
        reaction (cobra.Reaction): O objeto da reação.
        sp_map (dict): Mapa de metabolite_id para o seu shadow price.

    Returns:
        tuple: Uma tupla contendo (preço sombra médio, contagem de metabólitos).
    """
    shadow_prices_list = []
    
    # Itera sobre os metabólitos e suas estequiometrias na reação
    for met in reaction.metabolites:
        if met.id in sp_map:
            # Ponderação por estequiometria é uma opção para análise mais profunda,
            # mas para uma média simples, podemos usar a abordagem atual.
            # No entanto, a forma abaixo é mais robusta para lidar com a estequiometria.
            shadow_prices_list.append(sp_map[met.id] * abs(reaction.metabolites[met]))

    if not shadow_prices_list:
        return None, 0
    
    # A soma é dividida pela contagem total de metabólitos para obter a média
    # Opcionalmente, pode ser dividido pela soma das estequiometrias para uma média ponderada
    return sum(shadow_prices_list) / len(shadow_prices_list), len(shadow_prices_list)

def process_shadow_prices_by_reaction(input_excel_path, model_path, output_excel_path):
    """
    Função principal que orquestra o processamento dos preços sombra por reação,
    salvando os resultados em um novo arquivo Excel.
    """
    model = load_metabolic_model(model_path)
    if not model:
        return

    excel_sheets = load_shadow_price_data(input_excel_path)
    if not excel_sheets:
        return

    print("\n--- Iniciando a análise de preços sombra por reação ---")
    
    try:
        with pd.ExcelWriter(output_excel_path, engine='xlsxwriter') as writer:
            for sheet_name, df_shadow_prices in excel_sheets.items():
                print(f"Processando a condição de limitação: '{sheet_name}'")
                
                # Mapeia IDs de metabólitos para preços sombra para acesso rápido
                sp_map = pd.Series(df_shadow_prices['solver_sp'].values, index=df_shadow_prices['metabolite_id']).to_dict()

                reaction_data = []

                # Itera sobre todas as reações no modelo
                for reaction in model.reactions:
                    avg_sp, met_count = calculate_reaction_shadow_price(reaction, sp_map)
                    
                    if avg_sp is not None:
                        reaction_data.append({
                            'reaction_id': reaction.id,
                            'reaction_name': reaction.name,
                            'GPR': reaction.gene_reaction_rule,
                            'metabolite_names': " | ".join([met.name for met in reaction.metabolites]),
                            'average_shadow_price': avg_sp,
                            'metabolite_count': met_count
                        })
                
                df_output = pd.DataFrame(reaction_data)
                
                if df_output.empty:
                    print(f"Aviso: Nenhuma reação com preços sombra encontrados para a condição '{sheet_name}'.")
                    continue
                
                # Salva o DataFrame em uma nova aba com o nome da condição
                df_output.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"Resultados para '{sheet_name}' salvos em '{output_excel_path}'.")

    except Exception as e:
        print(f"Erro inesperado durante a escrita do arquivo: {e}", file=sys.stderr)

    print("\nAnálise completa e resultados exportados.")

if __name__ == "__main__":
    # Definindo os caminhos dos arquivos
    INPUT_EXCEL_PATH = 'analise_precos_sombra_condicoes_limitantes_v2.xlsx'
    MODEL_PATH = 'iJO1366.xml'
    OUTPUT_EXCEL_PATH = 'analise_precos_sombra_por_reacao.xlsx'
    
    # Execução principal do script
    process_shadow_prices_by_reaction(INPUT_EXCEL_PATH, MODEL_PATH, OUTPUT_EXCEL_PATH)