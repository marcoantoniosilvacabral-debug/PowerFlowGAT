import cobra
import json
import numpy as np
import warnings
import os
import pandas as pd
from scipy.linalg import pinv
from cobra.util import array
import ast
import re
import sys
from sympy import symbols, And, Or, sympify

import networkx as nx

from scipy import sparse
from scipy.stats import percentileofscore

import markov_clustering as mc

from matplotlib import pyplot as plt
from matplotlib import cm

# --- 0. Configurar o solver COBRApy (Global ou antes da criação da instância MFG) ---
try:
    cobra.Configuration().solver = 'glpk'
    print(f"Solver COBRApy configurado para: {cobra.Configuration().solver}")
except Exception as e:
    print(f"ERRO CRÍTICO: Não foi possível configurar o solver GLPK. Erro: {e}")
    print("Por favor, certifique-se de que o GLPK está instalado no seu sistema e que 'swiglpk' (ou outro binding) está instalado no seu ambiente Python.")
    print("Exemplo de instalação: pip install swiglpk")
    exit()

# --- FUNÇÕES DE ESSENCIALIDADE ADICIONADAS ---

def extract_genes(gpr_str):
    """
    Extrai todos os genes (locus tags) da regra GPR usando regex.
    :param gpr_str: String da regra GPR.
    :return: Set de genes únicos.
    """
    return set(re.findall(r'b\d{4}', gpr_str))

def does_deletion_deactivate(expr, gene_symbol, other_symbols):
    """
    Avalia o GPR booleano com o gene setado para False e outros para True.
    :param expr: Expressão SymPy do GPR.
    :param gene_symbol: Símbolo SymPy do gene a deletar.
    :param other_symbols: Lista de símbolos SymPy dos outros genes.
    :return: True se a expressão avalia para False (reação desativada).
    """
    subs = {gene_symbol: False}
    for other_sym in other_symbols:
        subs[other_sym] = True
    eval_expr = expr.subs(subs)
    return not eval_expr

def assign_reaction_labels(df_glucose, model):
    """
    Atribui labels de essencialidade às reações com base nos labels de genes e regras GPR,
    seguindo a lógica do FlowGAT (transferência de labels para reações desativadas por deleções).
    :param df_glucose: DataFrame filtrado para condição Glucose com gene_labels.
    :param model: Modelo COBRA carregado.
    :return: Dicionário com reaction_id como chave e essentiality_label como valor.
    """
    gene_to_label = df_glucose.set_index('Gene')['gene_label'].to_dict()
    
    # Coletar todos os genes únicos de todos os GPRs
    all_genes = set()
    for rxn in model.reactions:
        gpr = rxn.gene_reaction_rule.strip() if rxn.gene_reaction_rule else ''
        if gpr:
            all_genes.update(extract_genes(gpr))
    
    # Criar símbolos SymPy para genes
    gene_symbols = {gene: symbols(gene) for gene in all_genes}
    
    reaction_data = {}
    for rxn in model.reactions:
        rxn_id = rxn.id
        gpr = rxn.gene_reaction_rule.strip() if rxn.gene_reaction_rule else ''
        
        if not gpr:
            reaction_data[rxn_id] = -1
            continue
        
        # Preparar GPR para parsing SymPy
        gpr_parsed = gpr.replace(' and ', ' And ').replace(' or ', ' Or ')
        
        try:
            expr = sympify(gpr_parsed, evaluate=False)
        except Exception as e:
            print(f"Erro ao parsear GPR para {rxn_id}: {e}", file=sys.stderr)
            reaction_data[rxn_id] = -1
            continue
        
        # Genes na GPR
        genes_in_gpr = extract_genes(gpr)
        
        # Coletar labels de genes que desativam a reação
        deactivating_labels = []
        for gene in genes_in_gpr:
            if gene in gene_to_label:
                other_symbols = [gene_symbols[g] for g in genes_in_gpr if g != gene]
                if does_deletion_deactivate(expr, gene_symbols[gene], other_symbols):
                    deactivating_labels.append(gene_to_label[gene])
        
        # Atribuir label baseado nos deactivating genes (similar ao FlowGAT)
        if not deactivating_labels:
            label = -1  # Não mapeado ou many-to-many não resolvível
        else:
            unique_labels = set(deactivating_labels)
            if len(unique_labels) > 1:
                label = -1  # Conflito de labels
            elif 1 in unique_labels:
                label = 1  # Essencial se qualquer gene desativador for essencial
            else:
                label = 0  # Não essencial se todos desativadores forem não essenciais
        
        reaction_data[rxn_id] = label
    
    return reaction_data

def calculate_essentiality_regression_advanced(df_glucose, model):
    """
    Versão mais avançada que considera a lógica booleana do GPR
    para calcular valores de regression mais precisos.
    :param df_glucose: DataFrame com dados experimentais
    :param model: Modelo COBRA carregado
    :return: Dicionário com reaction_id como chave e essentiality_regression como valor
    """
    gene_to_growth = df_glucose.set_index('Gene')['Measured_growth'].to_dict()
    all_genes = set()
    
    for rxn in model.reactions:
        gpr = rxn.gene_reaction_rule.strip() if rxn.gene_reaction_rule else ''
        if gpr:
            all_genes.update(extract_genes(gpr))
    
    gene_symbols = {gene: symbols(gene) for gene in all_genes}
    reaction_regression = {}
    
    for rxn in model.reactions:
        rxn_id = rxn.id
        gpr = rxn.gene_reaction_rule.strip() if rxn.gene_reaction_rule else ''
        
        if not gpr:
            reaction_regression[rxn_id] = -1.0
            continue
        
        gpr_parsed = gpr.replace(' and ', ' And ').replace(' or ', ' Or ')
        
        try:
            expr = sympify(gpr_parsed, evaluate=False)
        except Exception:
            reaction_regression[rxn_id] = -1.0
            continue
        
        genes_in_gpr = extract_genes(gpr)
        available_genes = [g for g in genes_in_gpr if g in gene_to_growth]
        
        if not available_genes:
            reaction_regression[rxn_id] = -1.0
            continue
        
        # Calcular impacto baseado na lógica GPR
        if ' And ' in gpr:  # Complexo enzimático - todos necessários
            # Usar o pior cenário (menor crescimento)
            growth_values = [gene_to_growth[g] for g in available_genes]
            min_growth = min(growth_values)
            regression_value = 1.0 - min_growth
        
        elif ' or ' in gpr:  # Isoenzimas - qualquer um serve
            # Usar o melhor cenário (maior crescimento)  
            growth_values = [gene_to_growth[g] for g in available_genes]
            max_growth = max(growth_values)
            regression_value = 1.0 - max_growth
        
        else:  # Gene único
            growth_value = gene_to_growth[available_genes[0]]
            regression_value = 1.0 - growth_value
        
        # Garantir que está no range [-1, 1]
        regression_value = max(min(regression_value, 1.0), -1.0)
        reaction_regression[rxn_id] = regression_value
    
    return reaction_regression

def load_experimental_data():
    """
    Carrega e processa os dados experimentais do arquivo XLSX.
    :return: DataFrame filtrado para condição Glucose
    """
    try:
        df = pd.read_excel('Supplementary Data File 11.xlsx', sheet_name='Table 1')
    except FileNotFoundError:
        print("AVISO: Arquivo 'Supplementary Data File 11.xlsx' não encontrado. Pulando análise de essencialidade experimental.", file=sys.stderr)
        return None
    
    df_glucose = df[df['Condition'] == 'Glucose'].copy()
    
    if df_glucose.empty:
        print("AVISO: Nenhuma entrada encontrada para a condição 'Glucose'. Pulando análise de essencialidade experimental.", file=sys.stderr)
        return None
    
    # Binarizar labels com limiar do FlowGAT (0.5)
    df_glucose['gene_label'] = (df_glucose['Measured_growth'] < 0.5).astype(int)
    
    return df_glucose

# --- CLASSE MFG PRINCIPAL ---

class MFG():
    """Class representation for a mass flow graph
    Parameters
    ----------
    model_path : str
        Path to the genome-scale metabolic model file (e.g., 'iML1515_glucose.json').
    solution : cobra.Solution, optional
        A cobra Solution for building the MFG.
        [if not provided, it will be computed from the model with FBA]

    Attributes
    ----------
    model : cobra.Model
        The cobra model corresponding to the MFG instance.
    nodes : pandas.DataFrame
        A DataFrame of nodes of the Wild Type (WT) graph with columns: id, label, pagerank, betweenness.
    edges : pandas.DataFrame
        A DataFrame of the edges of the WT graph with columns: source, target, weight.
    matrix : numpy.array
        A numpy array of the adjacency matrix of the WT graph.
    solution : cobra.Solution
        The solution obtained from optimizing the WT model.
    v : numpy.array
        A numpy array of the flux vector of the solution.
    S2m_plus : numpy.array
        A numpy array of the stoichiometrix production matrix.
    S2m_minus : numpy.array
        A numpy array of the stoichiometrix consumption matrix.
    m : int
        An integer stating the number of reactions in the model.
    """
    def __init__(self, model_path, solution = None):

        # --- 1. Carregar o Modelo Metabólico ---
        print(f"\n--- Carregando o Modelo Metabólico a partir de '{model_path}' ---")
        try:
            self.model = cobra.io.load_json_model(model_path)
            print(f"Modelo de E. coli carregado com sucesso.")
            print(f"Número de reações no modelo: {len(self.model.reactions)}")
            print(f"Número de metabólitos no modelo: {len(self.model.metabolites)}")
        except FileNotFoundError:
            raise FileNotFoundError(f"ERRO CRÍTICO: O arquivo '{model_path}' não foi encontrado. "
                                     "Certifique-se de que o arquivo está no mesmo diretório do script ou forneça o caminho completo.")
        except Exception as e:
            raise Exception(f"ERRO CRÍTICO ao carregar o modelo: {e}")

        # --- 2. Aplicar as Condições de Contorno (Bounds) para FBA de "Wild Type" com Glicose ---
        OXYGEN_EXCHANGE_ID = "EX_o2_e"
        GLUCOSE_EXCHANGE_ID = "EX_glc__D_e"
        BIOMASS_REACTION_ID = 'BIOMASS_Ec_iML1515_core_75p37M'

        print("\n--- Configurando os limites das reações (bounds) para simulação de glicose como única fonte ---")

        ESSENTIAL_INORGANIC_NUTRIENTS = [
            "EX_nh4_e", "EX_pi_e", "EX_so4_e", "EX_h2o_e", "EX_h_e", "EX_fe2_e",
            "EX_mg2_e", "EX_ca2_e", "EX_cl_e", "EX_k_e", "EX_na1_e", "EX_ni2_e",
            "EX_cu2_e", "EX_mn2_e", "EX_mobd_e", "EX_cobalt2_e", "EX_zn2_e"
        ]

        for reaction in self.model.reactions:
            if reaction.id.startswith("EX_"):
                reaction.lower_bound = 0.0
                reaction.upper_bound = 1000.0

        try:
            reaction_o2 = self.model.reactions.get_by_id(OXYGEN_EXCHANGE_ID)
            reaction_o2.lower_bound = -20.0
            print(f"- Limite inferior de {OXYGEN_EXCHANGE_ID} (Oxigênio) ajustado para: {reaction_o2.lower_bound}.")
        except KeyError:
            warnings.warn(f"Reação '{OXYGEN_EXCHANGE_ID}' não encontrada no modelo. Verifique o ID.")

        try:
            reaction_glucose = self.model.reactions.get_by_id(GLUCOSE_EXCHANGE_ID)
            reaction_glucose.lower_bound = -10.0
            print(f"- Limite inferior de {GLUCOSE_EXCHANGE_ID} (D-Glicose) ajustado para: {reaction_glucose.lower_bound}.")
        except KeyError:
            raise KeyError(f"ERRO GRAVE: Reação '{GLUCOSE_EXCHANGE_ID}' (D-Glicose) NÃO ENCONTRADA no modelo. Verifique o ID e o arquivo JSON.")

        for nutrient_id in ESSENTIAL_INORGANIC_NUTRIENTS:
            try:
                nutrient_reaction = self.model.reactions.get_by_id(nutrient_id)
                nutrient_reaction.lower_bound = -1000.0
                nutrient_reaction.upper_bound = 1000.0
            except KeyError:
                warnings.warn(f"Reação de nutriente essencial '{nutrient_id}' (da sua lista ESSENTIAL_INORGANIC_NUTRIENTS) não encontrada no modelo. Verifique o ID ou se o modelo está completo.")

        ATPM_REACTION_ID = 'ATPM'
        ATPM_STANDARD_LOWER_BOUND = 6.86
        try:
            atpm_reaction = self.model.reactions.get_by_id(ATPM_REACTION_ID)
            original_atpm_lb = atpm_reaction.lower_bound
            atpm_reaction.lower_bound = ATPM_STANDARD_LOWER_BOUND
            print(f"- Reação de manutenção de ATP '{ATPM_REACTION_ID}' ajustada de {original_atpm_lb} para {atpm_reaction.lower_bound} (valor padrão/do artigo).")
        except KeyError:
            warnings.warn(f"Reação de manutenção de ATP '{ATPM_REACTION_ID}' não encontrada. Verifique o ID do modelo.")

        print("Limites das reações de troca e outras reações específicas configurados.")

        # --- 3. Definir a Função Objetivo ---
        print("\n--- Definindo a função objetivo ---")
        try:
            biomass_reaction = self.model.reactions.get_by_id(BIOMASS_REACTION_ID)
            self.model.objective = biomass_reaction
            self.model.objective.direction = 'max'
            print(f"- Função objetivo definida para: {biomass_reaction.id} (crescimento), direção: {self.model.objective.direction}.")
        except KeyError:
            raise KeyError(f"ERRO GRAVE: Reação de biomassa '{BIOMASS_REACTION_ID}' NÃO ENCONTRADA no modelo. Verifique o ID e o arquivo JSON.")
        except Exception as e:
            raise Exception(f"ERRO ao definir a função objetivo: {e}")

        # --- 4. Executar a FBA ---
        print("\n--- Executando Flux Balance Analysis (FBA) ---")
        self.solution = solution

        if self.solution is None:
            try:
                self.solution = self.model.optimize()
                if self.solution.status != 'optimal':
                    warnings.warn(f"FBA não retornou uma solução ótima. Status: {self.solution.status}")
                if self.solution.objective_value < 1e-6:
                    warnings.warn(f"FBA retornou um valor objetivo muito próximo de zero ({self.solution.objective_value:.6f}). Pode indicar crescimento nulo.")

                print("\n--- Resultado da FBA ---")
                print(f"Status da solução: {self.solution.status}")
                print(f"Taxa de crescimento (fluxo da função objetivo): {self.solution.objective_value:.6f}")

                print("\n--- Verificando Requisitos da Reação de Biomassa (Detalhado e Corrigido) ---")
                biomass_reaction = self.model.reactions.get_by_id(BIOMASS_REACTION_ID)

                required_metabolites = {}
                for met, coeff in biomass_reaction.metabolites.items():
                    if coeff < 0:
                        required_metabolites[met.id] = {'coefficient': coeff, 'compartment': met.compartment}

                print(f"Metabólitos CONSUMIDOS pela reação de biomassa '{BIOMASS_REACTION_ID}':")
                if not required_metabolites:
                    print("  (Nenhum metabólito consumidor encontrado na reação de biomassa. Isso é incomum.)")

                issues_found_in_biomass_check = False
                for met_id, data in required_metabolites.items():
                    coeff = data['coefficient']
                    compartment = data['compartment']

                    exchange_rxn_id_guess = None

                    if compartment in ['c', 'p']:
                        base_met_id = met_id
                        if base_met_id.endswith("_c") or base_met_id.endswith("_p"):
                            base_met_id = base_met_id[:-2]
                        exchange_rxn_id_guess = f"EX_{base_met_id}_e"

                        if met_id == 'h2o_c':
                            exchange_rxn_id_guess = 'EX_h2o_e'
                        elif met_id == 'h_c':
                            exchange_rxn_id_guess = 'EX_h_e'

                    elif compartment == 'e':
                        exchange_rxn_id_guess = f"EX_{met_id}"

                    if exchange_rxn_id_guess:
                        try:
                            exchange_reaction = self.model.reactions.get_by_id(exchange_rxn_id_guess)

                            if exchange_reaction.lower_bound >= -1e-9:
                                print(f"  ❌ PROBLEMA (ESPERADO EM MEIO MÍNIMO): Sua reação de troca '{exchange_rxn_id_guess}' tem lower_bound {exchange_reaction.lower_bound}.")
                                print("      Isso significa que o modelo NÃO PODE captar este nutriente externo. Em um meio mínimo, isso é normal, pois espera-se que o organismo o sintetize.")
                            else:
                                pass

                            flux_value = self.solution.fluxes.get(exchange_rxn_id_guess, None)
                            if flux_value is not None:
                                if flux_value > -1e-9 and exchange_reaction.lower_bound < -1e-9:
                                    if met_id not in ['h2o_c', 'h_c']:
                                        print(f"  AVISO: Embora aberta, o modelo NÃO ESTÁ CAPTANDO este nutriente '{exchange_rxn_id_guess}' (fluxo não negativo).")

                            else:
                                print(f"  AVISO: Fluxo para '{exchange_rxn_id_guess}' não encontrado na solução da FBA.")

                        except KeyError:
                            print(f"  ❌ PROBLEMA (ESPERADO EM MEIO MÍNIMO): Reação de troca esperada '{exchange_rxn_id_guess}' (para o ambiente externo) NÃO FOI ENCONTRADA no modelo.")
                            print(f"      O metabólito interno '{met_id}' (no compartimento '{compartment}') é exigido pela biomassa.")
                            print("      Isso é normal para metabólitos que devem ser sintetizados internamente e não captados do ambiente.")


                if not issues_found_in_biomass_check:
                    print("\n  Todas as vias para metabólitos exigidos pela biomassa parecem estar ativas ou são sintetizadas internamente (conforme esperado em meio mínimo).")
                    print("  Se a taxa de crescimento ainda é zero, verifique a manutenção de ATP ou outros limites de reações internas.")
                else:
                    print("\n  AVISO: FORAM ENCONTRADAS MENSAGENS SOBRE ACESSO A NUTRIENTES. No contexto de um meio mínimo, elas são geralmente informativas sobre a síntese interna, não problemas que impedem o crescimento se a FBA for bem-sucedida.")
                print("----------------------------------------------------------")

            except Exception as e:
                raise Exception(f"\nERRO FATAL ao executar a FBA: {e}. Verifique se o modelo está consistente e se a configuração de limites não torna o problema metabolicamente inviável.")

        if self.solution.status == 'optimal' and self.solution.objective_value > 1e-6:
            self.v = self.solution.fluxes.values.reshape([len(self.model.reactions),1])
            self.m = len(self.model.reactions)

            S = array.create_stoichiometric_matrix(self.model)
            m = S.shape[1]
            self.m = m

            rlist = [r.reversibility for r in self.model.reactions]
            r = np.zeros([m, 1])
            r[rlist] = 1
            R = np.diag(r[:,0])

            Im = np.identity(m)
            S2m_1 = np.block([S, -S])
            S2m_2 = np.block([[Im, np.zeros([m,m])], [np.zeros([m,m]), R]])
            self.S2m = S2m_1@S2m_2
            abs_S2m = np.abs(self.S2m)
            self.S2m_plus = 0.5 * (abs_S2m + self.S2m)
            self.S2m_minus = 0.5 * (abs_S2m - self.S2m)

            self.v2m = self.compute_v2m(self.v)
            self.matrix = self.compute_mfg(self.v)

            self.nodes, self.edges = self.compute_nodes_and_edges(self.matrix)

            # --- CALCULAR TODAS AS PREDIÇÕES DE ESSENCIALIDADE ---
            print("\n--- Calculando predições de essencialidade ---")
            
            # 1. Predições FBA (original)
            print("1. Calculando predições FBA com deleção manual...")
            self.fba_predictions = self._calculate_fba_predictions()
            self.nodes['fba_pred'] = self.nodes['label'].map(self.fba_predictions).fillna(0).astype(int)
            
            # 2. Essencialidade Experimental
            print("2. Calculando essencialidade experimental...")
            df_glucose = load_experimental_data()
            
            if df_glucose is not None:
                essentiality_labels = assign_reaction_labels(df_glucose, self.model)
                essentiality_regression = calculate_essentiality_regression_advanced(df_glucose, self.model)
                
                # Mapear apenas para reações que existem no MFG
                self.nodes['essentiality'] = self.nodes['label'].map(essentiality_labels).fillna(-1).astype(int)
                self.nodes['essentiality_regression'] = self.nodes['label'].map(essentiality_regression).fillna(-1.0)
                
                print(f"   - Essencialidade experimental calculada para {len(essentiality_labels)} reações")
                print(f"   - Regressão de essencialidade calculada para {len(essentiality_regression)} reações")
            else:
                # Se não há dados experimentais, preencher com valores padrão
                self.nodes['essentiality'] = -1
                self.nodes['essentiality_regression'] = -1.0
                print("   - Dados experimentais não disponíveis, preenchendo com valores padrão")

            self.nodes = self.centralities(self.nodes, self.edges)
            print("\nMass Flow Graph (MFG) construído e todas as predições de essencialidade calculadas.")
        else:
            raise Exception("Não foi possível construir o Mass Flow Graph porque a FBA não encontrou uma solução ótima com crescimento positivo. "
                            "Verifique as condições de contorno e a função objetivo.")
    
    # Nova função auxiliar para avaliar as regras de GPR
    def _evaluate_gpr(self, gpr_rule, knocked_out_genes):
        """
        Avalia se a GPR de uma reação é satisfeita com um conjunto de genes nocauteados.
        
        Parâmetros:
        gpr_rule (str): A regra de GPR como uma string booleana (e.g., '(gene1 and gene2) or gene3').
        knocked_out_genes (set): Um conjunto de IDs de genes nocauteados.

        Retorna:
        bool: True se a regra ainda for satisfeita, False caso contrário.
        """
        if not gpr_rule:
            return True # Reação sem GPR não é afetada

        # Substitui 'and' por '&' e 'or' por '|' para usar o avaliador de expressão
        gpr_expr = gpr_rule.replace(' and ', ' & ').replace(' or ', ' | ').replace('( ', '(').replace(' )', ')')
        
        # Cria um ambiente de avaliação para a expressão
        # Cada gene que não está nocauteado é True, e o nocauteado é False
        gene_values = {gene.id: (gene.id not in knocked_out_genes) for gene in self.model.genes}

        # Analisa a string da GPR para evitar a injeção de código malicioso
        try:
            tree = ast.parse(gpr_expr, mode='eval')
            # Avalia a árvore de sintaxe, substituindo nomes de genes por seus valores
            result = eval(compile(tree, '<string>', 'eval'), {}, gene_values)
            return result
        except (SyntaxError, NameError):
            # Se a GPR não for uma expressão booleana válida, assumimos que ela não é afetada
            return True


    def _calculate_fba_predictions(self):
        """
        Calcula predições de essencialidade para genes e reações através
        da deleção manual e da avaliação das GPRs.

        Retorna:
        --------
        dict
            Dicionário com IDs de reações como chaves e predições de essencialidade
            como valores (1 para essencial, -1 para não-essencial).
        """
        fba_predictions = {}
        
        ESSENTIALITY_THRESHOLD_PERCENT = 0.01
        wild_type_growth_rate = self.solution.objective_value
        
        # --- Prepara um dicionário de reações que são afetadas por cada gene ---
        gene_to_reactions = {gene.id: [] for gene in self.model.genes}
        for reaction in self.model.reactions:
            if reaction.gene_reaction_rule:
                genes_in_rule = {g.id for g in reaction.genes}
                for gene_id in genes_in_rule:
                    if gene_id in gene_to_reactions:
                        gene_to_reactions[gene_id].append(reaction.id)
        
        print("\n--- Realizando análise de nocaute de genes (manual com avaliação de GPR)... ---")
        for gene_to_knock_out in self.model.genes:
            knocked_out_genes_ids = {gene_to_knock_out.id}
            reactions_to_knock_out = []

            # Para cada reação, verificamos se ela é desativada por este nocaute
            for reaction in self.model.reactions:
                # Se a reação não tiver GPR, ela não é afetada por deleção de gene
                if not reaction.gene_reaction_rule:
                    continue

                if not self._evaluate_gpr(reaction.gene_reaction_rule, knocked_out_genes_ids):
                    reactions_to_knock_out.append(reaction)
            
            # Se o nocaute do gene não afeta nenhuma reação, pule para o próximo
            if not reactions_to_knock_out:
                if wild_type_growth_rate > 1e-9:
                     for reaction_id in gene_to_reactions.get(gene_to_knock_out.id, []):
                        fba_predictions[reaction_id] = -1
                continue

            with self.model:
                # Desativa as reações identificadas
                for reaction in reactions_to_knock_out:
                    reaction.knock_out()
                
                try:
                    ko_solution = self.model.optimize()
                    growth_rate_ko = ko_solution.objective_value if ko_solution.status == 'optimal' else 0
                    
                    if wild_type_growth_rate > 1e-9:
                        growth_drop = 1 - (growth_rate_ko / wild_type_growth_rate)
                    else:
                        growth_drop = 1

                    if growth_drop >= (1 - ESSENTIALITY_THRESHOLD_PERCENT):
                        prediction = 1
                    else:
                        prediction = -1
                except Exception as e:
                    print(f"Aviso: Erro ao otimizar após nocaute do gene '{gene_to_knock_out.id}'. Erro: {e}")
                    prediction = 1

                # Mapeia a predição para todas as reações controladas por este gene
                for reaction in reactions_to_knock_out:
                    fba_predictions[reaction.id] = prediction

        # A etapa de nocaute de reações sem genes é mantida
        print("\n--- Realizando análise de nocaute de reações (reaction knock-out) para reações sem GPR... ---")
        for reaction in self.model.reactions:
            if reaction.id in fba_predictions or reaction.id.startswith(("EX_", "DM_", "SK_", "ATPM")):
                if reaction.id not in fba_predictions:
                    fba_predictions[reaction.id] = -1
                continue
            
            with self.model:
                reaction.knock_out()
                try:
                    ko_solution = self.model.optimize()
                    growth_rate_ko = ko_solution.objective_value if ko_solution.status == 'optimal' else 0
                    
                    if wild_type_growth_rate > 1e-9:
                        growth_drop = 1 - (growth_rate_ko / wild_type_growth_rate)
                    else:
                        growth_drop = 1

                    if growth_drop >= (1 - ESSENTIALITY_THRESHOLD_PERCENT):
                        prediction = 1
                    else:
                        prediction = -1
                except Exception as e:
                    print(f"Aviso: Erro ao otimizar após nocaute da reação '{reaction.id}'. Erro: {e}")
                    prediction = 1 # Assume como essencial

            fba_predictions[reaction.id] = prediction

        return fba_predictions
        
    def compute_v2m(self, v):
        abs_v = np.abs(v)
        v_plus = 0.5 * (abs_v + v).T
        v_minus = 0.5 * (abs_v - v).T
        v2m = np.block([v_plus, v_minus]).T
        return v2m

    def compute_mfg(self, v):
        v2m = self.compute_v2m(v)
        V = np.diag(v2m[:,0])

        j_v = self.S2m_plus@v2m
        J_v = np.diag(j_v[:,0])

        inverse_J = pinv(J_v)
        mfg = (self.S2m_plus@V).T@inverse_J@(self.S2m_minus@V)
        return mfg

    def compute_nodes_and_edges(self, matrix):
        ids = np.arange(0,2*self.m).reshape([2*self.m,1])

        labels = np.array([r.id for r in self.model.reactions])
        labels = np.hstack([labels,labels]).reshape([2*self.m,1])

        EDGE_WEIGHT_TOLERANCE = 1.16e-10
        edgesarray = np.vstack((np.where(~np.isclose(matrix, 0.0, atol=EDGE_WEIGHT_TOLERANCE)), matrix[np.where(~np.isclose(matrix, 0.0, atol=EDGE_WEIGHT_TOLERANCE))])).T
        edges = pd.DataFrame(data = edgesarray).rename(columns={0:'Source', 1:'Target', 2:'Weight'})
        edges = edges.astype({'Source' : 'int', 'Target' : 'int'})

        active = np.unique(edgesarray[:,0:2].reshape(-1).astype('int'))
        nodes = pd.DataFrame(data=np.block([ids[active], labels[active]])).rename(columns={0:'id', 1:'label'})
        return nodes, edges


    def centralities(self, nodes, edges):
        G = nx.from_pandas_edgelist(edges, 'Source', 'Target')
        pr = nx.pagerank(G)

        prs = [pr[1] for pr in sorted(pr.items())]
        nodes['pagerank']=prs

        pr_sorted = sorted(prs)
        perc = nodes['pagerank'].apply(lambda x: percentileofscore(pr_sorted, x))
        nodes['pagerank percentile'] = perc

        bt = nx.betweenness_centrality(G)
        bts = [bt[1] for bt in sorted(bt.items())]
        nodes['betweenness']=bts
        return nodes

    def draw(self):
        plt.figure()
        G = nx.DiGraph()

        for s,t,w in zip(self.edges['Source'], self.edges['Target'], self.edges['Weight']):
            G.add_weighted_edges_from([(s,t, w)])

        pos = nx.spring_layout(G)

        nx.draw(G, pos = pos, node_size=20, with_labels=False, width=0.3)


    def cluster(self, draw = False):
        G = nx.DiGraph()

        for s,t,w in zip(self.edges['Source'], self.edges['Target'], self.edges['Weight']):
            G.add_weighted_edges_from([(s,t, w)])

        matrix = nx.to_scipy_sparse_matrix(G)
        result = mc.run_mcl(matrix)
        clusters = mc.get_clusters(result)

        if draw:
            plt.figure()
            pos = nx.spring_layout(G)
            tr = list(pos.keys())

            pos = [pos[tr[i]] for i in range(len(tr))]
            mc.draw_graph(matrix, clusters, pos = pos, node_size=20, with_labels=False,
                          cmap = cm.OrRd_r, width=0.3)
        return clusters

    def export(self, filename='mfg', directory='', matrix=True, nodes=True, edges=True):
        # --- Exportar matriz com índices dos nós essenciais ---
        if matrix:
            essential_nodes_df = self.nodes[self.nodes['fba_pred'] == 1]
            essential_nodes_indices = essential_nodes_df.index.values
            np.save(f'{directory}{filename}.npy', essential_nodes_indices)
            print(f"Arquivo '{filename}.npy' gerado com sucesso, contendo {len(essential_nodes_indices)} índices de nós essenciais.")

        # --- Exportar nós com TODAS as colunas na ordem especificada ---
        if nodes:
            # Adicionar coluna sem nome como contador de linhas
            nodes_to_export = self.nodes.copy()
            nodes_to_export.insert(0, '', range(1, len(nodes_to_export) + 1))
            
            # Ordem especificada: id,label,pagerank,pagerank percentile,betweenness,essentiality,fba_pred,essentiality_regression
            columns_order = ['', 'id', 'label', 'pagerank', 'pagerank percentile', 'betweenness', 'essentiality', 'fba_pred', 'essentiality_regression']
            
            # Verificar quais colunas existem no DataFrame
            existing_columns = [col for col in columns_order if col in nodes_to_export.columns]
            
            nodes_to_export = nodes_to_export[existing_columns].copy()

            # Formatar valores numéricos
            if 'pagerank' in nodes_to_export.columns:
                nodes_to_export['pagerank'] = nodes_to_export['pagerank'].apply(lambda x: f"{x:.9f}")
            if 'betweenness' in nodes_to_export.columns:
                nodes_to_export['betweenness'] = nodes_to_export['betweenness'].apply(lambda x: f"{x:.9f}")
            if 'pagerank percentile' in nodes_to_export.columns:
                nodes_to_export['pagerank percentile'] = nodes_to_export['pagerank percentile'].apply(lambda x: f"{x:.14f}")
            if 'essentiality_regression' in nodes_to_export.columns:
                # Formatação especial para essentiality_regression: -1.0 e 1.0 com apenas um zero
                def format_essentiality_regression(x):
                    try:
                        x_float = float(x)
                        if abs(x_float) == 1.0:
                            return f"{x_float:.1f}"
                        else:
                            return f"{x_float:.6f}".rstrip('0').rstrip('.')
                    except (ValueError, TypeError):
                        return str(x)
                
                nodes_to_export['essentiality_regression'] = nodes_to_export['essentiality_regression'].apply(format_essentiality_regression)

            # Exportar sem índice para evitar a coluna "Unnamed"
            nodes_to_export.to_csv(f'{directory}{filename}_nodes.csv', index=False)
            print(f"Arquivo '{filename}_nodes.csv' gerado com {len(nodes_to_export)} nós e colunas: {', '.join(existing_columns)}")

        # --- Exportar arestas (inalterado) ---
        if edges:
            formatted_edges = self.edges.copy()
            formatted_edges['Weight'] = formatted_edges['Weight'].apply(
                lambda x: f"{x:.2E}" if (abs(x) < 1e-3 and x != 0) else f"{x:.9f}".rstrip('0').rstrip('.')
            )
            formatted_edges.to_csv(f'{directory}{filename}_edges.csv', index=False)
            print(f"Arquivo '{filename}_edges.csv' gerado com {len(formatted_edges)} arestas")


def gera_mfg(model, solution):
    """
    Gera um Mass Flow Graph (MFG) como um grafo NetworkX a partir de um modelo e solução FBA.

    Parâmetros:
    -----------
    model : cobra.Model
        Modelo metabólico carregado
    solution : cobra.Solution
        Solução da FBA

    Retorna:
    --------
    networkx.DiGraph
        Grafo direcionado representando o MFG
    """
    # Cria um arquivo temporário JSON (a classe MFG do seu código espera um arquivo)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmpfile:
        cobra.io.save_json_model(model, tmpfile.name)
        mfg_instance = MFG(tmpfile.name, solution)
        os.remove(tmpfile.name)  # Limpa o arquivo temporário

    # Converte para grafo NetworkX
    G = nx.DiGraph()

    # Adiciona nós
    for _, row in mfg_instance.nodes.iterrows():
        G.add_node(row['id'], label=row['label'])

    # Adiciona arestas com pesos
    for _, row in mfg_instance.edges.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row['Weight'])

    return G


# --- Exemplo de Uso ---
if __name__ == "__main__":
    model_file = 'iML1515_glucose.json'

    print("\n--- Iniciando o processo de construção do MFG unificado ---")
    try:
        mfg_instance = MFG(model_file)

        print("\n--- Análise básica do MFG unificado ---")
        print(f"Número de nós no MFG: {len(mfg_instance.nodes)}")
        print(f"Número de arestas no MFG: {len(mfg_instance.edges)}")
        print("\nPrimeiras 5 linhas dos nós (com todas as predições):")
        print(mfg_instance.nodes[['id', 'label', 'pagerank', 'betweenness', 'essentiality', 'fba_pred', 'essentiality_regression']].head())
        print("\nPrimeiras 5 linhas das arestas:")
        print(mfg_instance.edges.head())

        print("\n--- Exportando o MFG unificado para arquivos ---")
        mfg_instance.export(filename='mfg_with_manual_predictions', nodes=True, edges=True, matrix=True)
        print("Exportação concluída. Verifique os arquivos:")
        print("  - 'mfg_with_manual_predictions_nodes.csv' (com essentiality e essentiality_regression)")
        print("  - 'mfg_with_manual_predictions_edges.csv'")
        print("  - 'mfg_with_manual_predictions.npy'")

        # Estatísticas das predições
        if 'essentiality' in mfg_instance.nodes.columns:
            print(f"\nDistribuição de essentiality:")
            print(mfg_instance.nodes['essentiality'].value_counts().sort_index())
        if 'essentiality_regression' in mfg_instance.nodes.columns:
            print(f"\nDistribuição de essentiality_regression:")
            print(mfg_instance.nodes['essentiality_regression'].value_counts().sort_index())

    except Exception as e:
        print(f"\nOcorreu um erro durante a execução: {e}")