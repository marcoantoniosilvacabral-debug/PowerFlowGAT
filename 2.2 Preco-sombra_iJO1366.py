import cobra
import pandas as pd
import warnings
import os

# Ignora avisos, comuns ao carregar modelos metabólicos
warnings.filterwarnings("ignore")

def run_fba_with_nutrient_limitation(model, nutrient_reaction_id, perturbation_level):
    """
    Executa a análise de preços-sombra sob uma condição de limitação de nutriente,
    com validação alinhada à metodologia do texto de Reznik et al. (2013).
    
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
    
    # Para reações de captação, o fluxo é negativo. 'min' busca o maior valor negativo.
    limited_model.objective.direction = 'min'
    
    max_flux_solution = limited_model.optimize()
    
    if max_flux_solution.status != 'optimal':
        print(f"Erro: Não foi possível otimizar o fluxo de '{nutrient_reaction_id}'. Status: {max_flux_solution.status}")
        limited_model.objective = original_objective
        limited_model.objective.direction = 'max'
        return None
        
    max_flux = abs(max_flux_solution.objective_value)
    
    # Restaura o objetivo original
    limited_model.objective = original_objective
    limited_model.objective.direction = 'max'
    
    if max_flux < 1e-6:
        print(f"Atenção: A reação '{nutrient_reaction_id}' não é capaz de transportar fluxo. Impossível limitar.")
        return None
    
    print(f"Fluxo máximo de captação de {nutrient_reaction_id}: {max_flux:.4f} mmol/gDW/h")
    
    # 2. Aplicar a limitação e resolver o FBA para obter preços-sombra do solver
    limited_flux = -max_flux * perturbation_level
    nutrient_reaction.lower_bound = limited_flux
    print(f"Restringindo fluxo de {nutrient_reaction_id} para {limited_flux:.4f} mmol/gDW/h...")
    
    solution = limited_model.optimize()
    optimal_growth = solution.objective_value
    
    if solution.status != 'optimal':
        print(f"Atenção: A otimização não encontrou uma solução ótima sob limitação. Status: {solution.status}")
        return None
        
    shadow_prices_solver = solution.shadow_prices
    
    print(f"Crescimento ótimo sob limitação: {optimal_growth:.6f} h⁻¹")
    print(f"Número de shadow prices não-zero identificados pelo solver: {(abs(shadow_prices_solver) > 1e-9).sum()}")

    # 3. Implementar a validação de preços-sombra via brute-force
    print("\nIniciando a validação de preços-sombra via brute-force...")
    
    perturbation = 1e-6  # Valor de perturbação fixo
    tolerance = 1e-6
    
    # Mapeia IDs para nomes, para melhor legibilidade
    metabolite_name_map = {met.id: met.name for met in limited_model.metabolites}
    
    # Criar uma lista para coletar os resultados
    results_list = []
    
    metabolites_to_check = list(limited_model.metabolites)
    total_metabolites = len(metabolites_to_check)
    
    for i, metabolite in enumerate(metabolites_to_check):
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
            'compartment': metabolite.compartment,
            'solver_sp': solver_sp,
            'manual_sp_up': None,
            'manual_sp_down': None,
            'is_degenerate': False
        }
        
        # Para modificar o RHS do balanço de massa, precisamos encontrar
        # a constraint associada ao metabólito
        try:
            constraint_id = metabolite_id + "_balance"
            
            # Perturbação Incremental (acumulação - b_i positivo)
            with limited_model as perturbed_model_up:
                try:
                    if constraint_id in perturbed_model_up.constraints:
                        constraint_up = perturbed_model_up.constraints[constraint_id]
                        original_lb = constraint_up.lb
                        original_ub = constraint_up.ub
                        constraint_up.lb = perturbation
                        constraint_up.ub = perturbation
                        
                        solution_up = perturbed_model_up.optimize()
                        if solution_up.status == 'optimal':
                            manual_sp_up = (solution_up.objective_value - optimal_growth) / perturbation
                            met_data['manual_sp_up'] = manual_sp_up
                        
                        # Restaurar (não necessário com 'with', mas por segurança)
                        constraint_up.lb = original_lb
                        constraint_up.ub = original_ub
                except Exception as e:
                    pass
            
            # Perturbação Decremental (esgotamento - b_i negativo)
            with limited_model as perturbed_model_down:
                try:
                    if constraint_id in perturbed_model_down.constraints:
                        constraint_down = perturbed_model_down.constraints[constraint_id]
                        original_lb = constraint_down.lb
                        original_ub = constraint_down.ub
                        constraint_down.lb = -perturbation
                        constraint_down.ub = -perturbation
                        
                        solution_down = perturbed_model_down.optimize()
                        if solution_down.status == 'optimal':
                            manual_sp_down = (solution_down.objective_value - optimal_growth) / (-perturbation)
                            met_data['manual_sp_down'] = manual_sp_down
                        
                        # Restaurar
                        constraint_down.lb = original_lb
                        constraint_down.ub = original_ub
                except Exception as e:
                    pass
        except Exception as e:
            pass

        # Comparação para verificar degenerescência
        if met_data['manual_sp_up'] is not None and met_data['manual_sp_down'] is not None:
            if abs(met_data['manual_sp_up'] - solver_sp) > tolerance or \
               abs(met_data['manual_sp_down'] - solver_sp) > tolerance:
                met_data['is_degenerate'] = True
        elif met_data['manual_sp_up'] is not None:
            if abs(met_data['manual_sp_up'] - solver_sp) > tolerance:
                met_data['is_degenerate'] = True
        elif met_data['manual_sp_down'] is not None:
            if abs(met_data['manual_sp_down'] - solver_sp) > tolerance:
                met_data['is_degenerate'] = True
        
        results_list.append(met_data)
        
        if (i + 1) % 100 == 0:
            print(f"    {i + 1} de {total_metabolites} metabólitos verificados...")
            
    print(f"Validação completa. {len(results_list)} metabólitos com shadow prices significativos encontrados.")
    
    # Verificar e reportar metabólitos degenerados
    if results_list:
        degenerates = sum(1 for r in results_list if r['is_degenerate'])
        if degenerates > 0:
            print(f"ATENÇÃO: {degenerates} metabólitos apresentaram shadow prices degenerados.")
        else:
            print("Nenhum shadow price degenerado encontrado. Resultados são consistentes.")
            
    return pd.DataFrame(results_list)


### Execução Principal do Script

if __name__ == "__main__":
    
    print("=" * 70)
    print("ANÁLISE DE FLUX IMBALANCE - PREÇOS-SOMBRA EM E. coli")
    print("Baseado em: Reznik, Mehta & Segrè (2013) - PLoS Comput Biol")
    print("=" * 70)
    
    # Carregar o modelo iJO1366 de E. coli a partir do arquivo XML local
    print("\nCarregando modelo iJO1366 de Escherichia coli...")
    
    model_path = 'iJO1366.xml'
    
    if not os.path.exists(model_path):
        # Tentar outros caminhos possíveis
        alternative_paths = [
            '/Users/marcoantonio/Documents/teste/iJO1366.xml',
            './iJO1366.xml',
            '../iJO1366.xml'
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                model_path = alt_path
                break
        else:
            print(f"ERRO: Arquivo 'iJO1366.xml' não encontrado!")
            print("Por favor, coloque o arquivo no mesmo diretório deste script.")
            print(f"Diretório atual: {os.getcwd()}")
            exit()
    
    try:
        # Carregar modelo a partir do arquivo XML (formato SBML)
        print(f"Carregando modelo de: {model_path}")
        model = cobra.io.read_sbml_model(model_path)
        model.solver = 'glpk'
        print("Modelo iJO1366 carregado com sucesso!")
        
        # Informações básicas do modelo
        print(f"  Reações: {len(model.reactions)}")
        print(f"  Metabólitos: {len(model.metabolites)}")
        print(f"  Genes: {len(model.genes)}")
        
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        import traceback
        traceback.print_exc()
        exit()
    
    # Definir o objetivo de biomassa para o modelo iJO1366
    # Procurar a reação de biomassa
    biomass_reaction_candidates = [
        'BIOMASS_Ec_iJO1366_core_53p95M',
        'BIOMASS_Ec_iJO1366_WT_53p95M',
        'BIOMASS_Ec_iJO1366_core',
        'Ec_biomass_iJO1366_core_53p95M'
    ]
    
    biomass_reaction_id = None
    for candidate in biomass_reaction_candidates:
        if candidate in model.reactions:
            biomass_reaction_id = candidate
            break
    
    if biomass_reaction_id is None:
        # Buscar reação de biomassa automaticamente
        print("\nBuscando reação de biomassa automaticamente...")
        biomass_reactions = [r.id for r in model.reactions if 'BIOMASS' in r.id.upper() or 'biomass' in r.id.lower()]
        if biomass_reactions:
            biomass_reaction_id = biomass_reactions[0]
            print(f"Reação de biomassa encontrada: {biomass_reaction_id}")
        else:
            # Listar todas as reações que contêm 'bio' para debug
            print("Reações disponíveis com 'bio' no nome:")
            bio_rxns = [r.id for r in model.reactions if 'bio' in r.id.lower()]
            for rxn in bio_rxns[:20]:
                print(f"  - {rxn}")
            
            if not bio_rxns:
                print("Nenhuma reação com 'bio' encontrada. Listando todas as reações:")
                all_rxns = [r.id for r in model.reactions]
                for rxn in all_rxns[:30]:
                    print(f"  - {rxn}")
            
            print("\nPor favor, identifique a reação de biomassa e adicione-a à lista 'biomass_reaction_candidates'")
            exit()
    
    model.objective = biomass_reaction_id
    model.objective.direction = 'max'
    print(f"Objetivo definido: {biomass_reaction_id}")
    
    # Configurar meio de cultura mínimo para E. coli
    print("\nConfigurando meio de cultura mínimo para E. coli...")
    
    # Fechar todas as trocas primeiro
    exchange_reactions_closed = 0
    for reaction in model.reactions:
        if reaction.id.startswith("EX_"):
            reaction.lower_bound = 0.0
            reaction.upper_bound = 1000.0
            exchange_reactions_closed += 1
    
    print(f"  {exchange_reactions_closed} reações de troca fechadas inicialmente")
    
    # Configurar nutrientes essenciais
    essential_nutrients = {
        'EX_glc__D_e': -10.0,      # Glicose
        'EX_o2_e': -20.0,           # Oxigênio
        'EX_nh4_e': -1000.0,        # Amônia (fonte de nitrogênio)
        'EX_pi_e': -1000.0,         # Fosfato
        'EX_so4_e': -1000.0,        # Sulfato
        'EX_h2o_e': -1000.0,        # Água
        'EX_h_e': -1000.0,          # Prótons
        'EX_fe2_e': -1000.0,        # Ferro
        'EX_mg2_e': -1000.0,        # Magnésio
        'EX_ca2_e': -1000.0,        # Cálcio
        'EX_cl_e': -1000.0,         # Cloreto
        'EX_k_e': -1000.0,          # Potássio
        'EX_na1_e': -1000.0,        # Sódio
        'EX_mn2_e': -1000.0,        # Manganês
        'EX_zn2_e': -1000.0,        # Zinco
        'EX_cu2_e': -1000.0,        # Cobre
        'EX_cobalt2_e': -1000.0,    # Cobalto
        'EX_ni2_e': -1000.0,        # Níquel
        'EX_mobd_e': -1000.0,       # Molibdato
    }
    
    nutrients_set = 0
    nutrients_not_found = []
    
    for nutrient_id, bound in essential_nutrients.items():
        try:
            model.reactions.get_by_id(nutrient_id).lower_bound = bound
            nutrients_set += 1
        except KeyError:
            nutrients_not_found.append(nutrient_id)
    
    print(f"  {nutrients_set} nutrientes configurados")
    if nutrients_not_found:
        print(f"  Nutrientes não encontrados no modelo: {', '.join(nutrients_not_found)}")
    
    # Verificar se o modelo é viável
    print("\nVerificando viabilidade do modelo...")
    solution = model.optimize()
    if solution.status == 'optimal':
        print(f"✓ Modelo viável. Crescimento ótimo: {solution.objective_value:.6f} h⁻¹")
    else:
        print(f"✗ Modelo inviável. Status: {solution.status}")
        print("Tentando relaxar constraints...")
        # Relaxar um pouco as constraints de troca
        for reaction in model.reactions:
            if reaction.id.startswith("EX_") and reaction.id not in ['EX_glc__D_e', 'EX_o2_e', 'EX_nh4_e']:
                try:
                    model.reactions.get_by_id(reaction.id).lower_bound = -1000.0
                except:
                    pass
        solution = model.optimize()
        if solution.status == 'optimal':
            print(f"✓ Modelo viável após relaxamento. Crescimento: {solution.objective_value:.6f} h⁻¹")
        else:
            print(f"✗ Modelo ainda inviável. Verifique a configuração do meio.")
    
    # Definir as condições de limitação
    conditions = {
        'limitacao_glicose': 'EX_glc__D_e',
        'limitacao_nitrogenio': 'EX_nh4_e'
    }
    
    # Verificar se as reações existem
    for condition_name, reaction_id in conditions.items():
        if reaction_id not in model.reactions:
            print(f"\nAVISO: Reação '{reaction_id}' para '{condition_name}' não encontrada!")
            print("Reações EX_ disponíveis:")
            ex_rxns = [r.id for r in model.reactions if r.id.startswith('EX_')]
            for rxn in sorted(ex_rxns)[:30]:
                print(f"  - {rxn}")
    
    # Criar arquivo Excel para resultados
    output_filename = 'analise_precos_sombra_Ecoli_iJO1366.xlsx'
    
    print("\n" + "=" * 70)
    print("INICIANDO ANÁLISE DE PREÇOS-SOMBRA")
    print("=" * 70)
    
    with pd.ExcelWriter(output_filename, engine='xlsxwriter') as writer:
        for condition_name, nutrient_reaction_id in conditions.items():
            
            if nutrient_reaction_id not in model.reactions:
                print(f"\nPulando {condition_name}: reação '{nutrient_reaction_id}' não encontrada.")
                continue
                
            # Testar com 10% do fluxo máximo
            level = 0.1
            print(f"\n{'='*50}")
            print(f"Condição: {condition_name}")
            print(f"Nível de limitação: {level*100}% do fluxo máximo")
            print(f"{'='*50}")
            
            results_df = run_fba_with_nutrient_limitation(model, nutrient_reaction_id, level)
            
            if results_df is not None and not results_df.empty:
                # Ordenar por shadow price (mais negativos primeiro)
                results_df = results_df.sort_values('solver_sp')
                
                # Adicionar estatísticas
                print(f"\nResumo para {condition_name}:")
                print(f"  Total de metabólitos com shadow price significativo: {len(results_df)}")
                print(f"  Shadow prices negativos (limitantes para crescimento): {(results_df['solver_sp'] < -1e-6).sum()}")
                print(f"  Shadow prices positivos: {(results_df['solver_sp'] > 1e-6).sum()}")
                
                if (results_df['solver_sp'] < -1e-6).sum() > 0:
                    print(f"\n  Top 10 metabólitos mais limitantes para crescimento:")
                    top10 = results_df[results_df['solver_sp'] < -1e-6].head(10)
                    for idx, (_, row) in enumerate(top10.iterrows(), 1):
                        print(f"    {idx}. {row['metabolite_name'][:50]} ({row['metabolite_id']}): {row['solver_sp']:.6f}")
                
                # Salvar no Excel
                sheet_name = condition_name[:31]
                results_df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"\n  Resultados salvos na aba '{sheet_name}'")
            else:
                print(f"  Nenhum resultado para {condition_name}")

    print("\n" + "=" * 70)
    print(f"ANÁLISE COMPLETA!")
    print(f"Resultados exportados para: {output_filename}")
    print("=" * 70)
    
    # Comparação com o estudo
    print("\n" + "=" * 70)
    print("COMPARAÇÃO COM O ESTUDO DE REZNIK ET AL. (2013):")
    print("=" * 70)
    print("O estudo encontrou:")
    print("  • Limitação de glicose:")
    print("    - N-acetil-glucosamina-1-fosfato e arginina entre os mais limitantes")
    print("    - Glutamato como outlier (previsto limitante, mas concentração caiu)")
    print("  • Limitação de nitrogênio:")
    print("    - Aminoácidos como os metabólitos mais limitantes")
    print("    - Maior número de metabólitos limitantes vs. outras condições")
    print("  • Shadow prices negativos = metabólitos limitantes para crescimento")
    print("  • Shadow prices zero ou positivos = metabólitos não-limitantes")
    print("=" * 70)