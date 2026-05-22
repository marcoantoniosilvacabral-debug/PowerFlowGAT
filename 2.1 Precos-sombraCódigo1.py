import cobra
import pandas as pd
import warnings
import os

# Ignora avisos, comuns ao carregar modelos metabólicos
warnings.filterwarnings("ignore")

def run_fba_with_nutrient_limitation(model, nutrient_reaction_id, perturbation_level):
    """
    Executa a análise de preços-sombra sob uma condição de limitação de nutriente,
    com validação alinhada à metodologia do texto.
    
    Args:
        model (cobra.Model): O modelo metabólico a ser analisado.
        nutrient_reaction_id (str): A ID da reação de transporte do nutriente a ser limitado.
        perturbation_level (float): O nível de limitação, expresso como uma porcentagem
                                    do fluxo máximo (ex: 0.1 para 10%).
    
    Returns:
        pandas.DataFrame: DataFrame contendo os resultados da análise de preços-sombra.
    """
    print(f"\n--- Analisando o modelo sob limitação de {nutrient_reaction_id.replace('EX_', '')} ---")
    
    # Trabalha com uma cópia para evitar modificações no modelo original
    limited_model = model.copy()
    
    # Encontra a reação de transporte do nutriente
    try:
        nutrient_reaction = limited_model.reactions.get_by_id(nutrient_reaction_id)
    except KeyError:
        print(f"Erro: Reação de nutriente '{nutrient_reaction_id}' não encontrada no modelo.")
        return None

    # 1. Obter o fluxo máximo do nutriente isoladamente
    original_objective = limited_model.objective
    limited_model.objective = nutrient_reaction
    
    # --- CORREÇÃO AQUI: AJUSTAR A DIREÇÃO DO OBJETIVO PARA ENCONTRAR O FLUXO MÁXIMO DE CAPTAÇÃO ---
    # Para reações de captação, o fluxo é negativo. 'min' busca o maior valor negativo.
    limited_model.objective.direction = 'min'
    
    max_flux_solution = limited_model.optimize()
    max_flux = abs(max_flux_solution.objective_value)
    
    # Restaura o objetivo original
    limited_model.objective = original_objective
    limited_model.objective.direction = 'max'
    
    if max_flux < 1e-6:
        print(f"Atenção: A reação '{nutrient_reaction_id}' não é capaz de transportar fluxo. Impossível limitar.")
        return None
    
    # 2. Aplicar a limitação e resolver o FBA para obter preços-sombra do solver
    # O fluxo é limitado no lado de captação (absorção), que tem lower_bound negativo
    limited_flux = -max_flux * perturbation_level
    nutrient_reaction.lower_bound = limited_flux
    print(f"2. Restringindo fluxo de {nutrient_reaction_id} para {limited_flux:.4f}...")
    
    solution = limited_model.optimize()
    optimal_growth = solution.objective_value
    
    if solution.status != 'optimal':
        print(f"Atenção: A otimização não encontrou uma solução ótima sob limitação. Status: {solution.status}")
        return None
        
    shadow_prices_solver = solution.shadow_prices
    
    print(f"Crescimento ótimo sob limitação: {optimal_growth:.6f}")

    # 3. Implementar a validação de preços-sombra via brute-force
    print("\n3. Iniciando a validação de preços-sombra via brute-force...")
    
    perturbation = 1e-6  # Valor de perturbação fixo
    tolerance = 1e-6
    
    # Mapeia IDs para nomes, para melhor legibilidade
    metabolite_name_map = {met.id: met.name for met in limited_model.metabolites}
    
    # Criar um DataFrame para coletar os resultados antes de salvar
    results_list = []
    
    for i, metabolite in enumerate(limited_model.metabolites):
        metabolite_id = metabolite.id
        
        # Ignorar metabólitos sem shadow price
        if metabolite_id not in shadow_prices_solver.index:
            continue
            
        solver_sp = shadow_prices_solver[metabolite_id]
        
        # Filtra shadow prices muito próximos de zero
        if abs(solver_sp) < 1e-9:
             continue
             
        # Dicionário para armazenar dados do metabólito atual
        met_data = {
            'metabolite_id': metabolite_id,
            'metabolite_name': metabolite_name_map.get(metabolite_id, "Nome Não Encontrado"),
            'solver_sp': solver_sp,
            'manual_sp_up': None,
            'manual_sp_down': None,
            'is_degenerate': False
        }
        
        # Perturbação Incremental (acumulação)
        with limited_model as perturbed_model_up:
            try:
                # Altera o lado direito do balanço de fluxo (bᵢ)
                perturbed_model_up.metabolites.get_by_id(metabolite_id)._model_reaction_matrix_column_for(metabolite).RHS = perturbation
                
                solution_up = perturbed_model_up.optimize()
                manual_sp_up = (solution_up.objective_value - optimal_growth) / perturbation
                met_data['manual_sp_up'] = manual_sp_up
            except Exception:
                pass
        
        # Perturbação Decremental (esgotamento)
        with limited_model as perturbed_model_down:
            try:
                perturbed_model_down.metabolites.get_by_id(metabolite_id)._model_reaction_matrix_column_for(metabolite).RHS = -perturbation
                
                solution_down = perturbed_model_down.optimize()
                manual_sp_down = (solution_down.objective_value - optimal_growth) / (-perturbation)
                met_data['manual_sp_down'] = manual_sp_down
            except Exception:
                pass

        # Comparação para verificar degenerescência
        if met_data['manual_sp_up'] is not None and met_data['manual_sp_down'] is not None:
            if abs(met_data['manual_sp_up'] - solver_sp) > tolerance or \
               abs(met_data['manual_sp_down'] - solver_sp) > tolerance:
                met_data['is_degenerate'] = True
        
        results_list.append(met_data)
        
        if (i + 1) % 50 == 0:
            print(f"    {i + 1} de {len(limited_model.metabolites)} metabólitos verificados...")
            
    return pd.DataFrame(results_list)


### Execução Principal do Script

if __name__ == "__main__":
    model_file = 'iML1515_glucose.json'
    
    # Carregar o modelo uma única vez
    try:
        model = cobra.io.load_json_model(model_file)
        model.solver = 'glpk'
        
        # Definir o objetivo de biomassa para o modelo iML1515
        biomass_reaction_id = 'BIOMASS_Ec_iML1515_core_75p37M'
        if biomass_reaction_id not in model.reactions:
            raise KeyError(f"Reação de biomassa '{biomass_reaction_id}' não encontrada no modelo.")
            
        model.objective = biomass_reaction_id
        
        # Ajustar os limites iniciais conforme o código do MFG
        EXCHANGE_REACTIONS = ["EX_o2_e", "EX_glc__D_e", "EX_nh4_e", "EX_pi_e", "EX_so4_e", "EX_h2o_e", "EX_h_e", "EX_fe2_e", "EX_mg2_e", "EX_ca2_e", "EX_cl_e", "EX_k_e", "EX_na1_e", "EX_ni2_e", "EX_cu2_e", "EX_mn2_e", "EX_mobd_e", "EX_cobalt2_e", "EX_zn2_e"]
        
        for reaction in model.reactions:
            if reaction.id.startswith("EX_"):
                reaction.lower_bound = 0.0
                reaction.upper_bound = 1000.0

        model.reactions.get_by_id("EX_o2_e").lower_bound = -20.0
        model.reactions.get_by_id("EX_glc__D_e").lower_bound = -10.0
        for nutrient_id in EXCHANGE_REACTIONS:
            try:
                if nutrient_id not in ["EX_o2_e", "EX_glc__D_e"]:
                    model.reactions.get_by_id(nutrient_id).lower_bound = -1000.0
            except KeyError:
                pass
        
    except Exception as e:
        print(f"Erro ao carregar o modelo ou definir o objetivo: {e}")
        exit()

    # Definir as condições de limitação para o modelo iML1515
    conditions = {
        'limitação_glicose': 'EX_glc__D_e',
        'limitação_oxigenio': 'EX_o2_e'
    }

    # Criar um objeto ExcelWriter para salvar em abas diferentes
    output_filename = 'analise_precos_sombra_iML1515.xlsx'
    with pd.ExcelWriter(output_filename, engine='xlsxwriter') as writer:
        for condition_name, nutrient_reaction_id in conditions.items():
            results_df = run_fba_with_nutrient_limitation(model, nutrient_reaction_id, 0.1)
            if results_df is not None:
                # Salva o DataFrame em uma nova aba com o nome da condição
                results_df.to_excel(writer, sheet_name=condition_name, index=False)
                print(f"Resultados para '{condition_name}' salvos com sucesso na aba '{condition_name}'.")

    print(f"\nAnálise completa. Todos os resultados foram exportados para '{output_filename}'.")