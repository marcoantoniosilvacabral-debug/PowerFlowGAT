import os
import sys
import warnings
from math import isnan
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict
import cobra
from cobra.io import load_json_model
from cobra.flux_analysis import flux_variability_analysis
from cobra import Reaction

class CONTRABASS_EROG:
    """
    Implementação FIDELIDADE ao artigo CONTRABASS para c=1 (EROG) com extensão RGR.
    Baseado estritamente em: "CONTRABASS: exploiting flux constraints in genome-scale models"
    
    Definições CORRETAS do artigo:
    - EROG = ER₁ = Essential Reactions for Optimal Growth (c=1)
    - Uma reação é EROG se knockout resulta em crescimento < lmax 
      (ou inviável) com restrição c=1 — sem threshold artificial
    """

    def __init__(self, model_path: str, existing_csv_path: str):
        self.model_path = model_path
        self.existing_csv_path = existing_csv_path
        self.CONST_EPSILON = 1e-6           # tolerância numérica para comparar fluxos/crescimento
        self.model = None
        self.lmax = 0.0
        self.erog_results = {}              # {reaction_id: {...}}
        self.fva_result_c1 = None
        self.chokepoints_c1 = set()
        self.nr_c1 = set()
        self.dr_c1 = set()
        self.rr_c1 = set()
        self.validation_stats = {}

    def load_and_validate_data(self) -> bool:
        """Carrega e valida ambos os datasets."""
        if not os.path.exists(self.existing_csv_path):
            raise FileNotFoundError(f"Dataset não encontrado: {self.existing_csv_path}")

        self.existing_df = pd.read_csv(self.existing_csv_path)
        print(f"Dataset existente carregado: {len(self.existing_df)} linhas")

        # Remover coluna EROG antiga se existir
        if 'EROG' in self.existing_df.columns:
            print("Coluna EROG existente removida (será recalculada fiel ao artigo)")
            self.existing_df = self.existing_df.drop('EROG', axis=1)

        required_cols = ['id', 'label', 'essentiality', 'fba_pred', 'shadowprice']
        missing_cols = [col for col in required_cols if col not in self.existing_df.columns]
        if missing_cols:
            raise ValueError(f"Colunas faltantes no dataset: {missing_cols}")

        print("\nCarregando modelo metabólico...")
        self.model = load_json_model(self.model_path)
        self._apply_model_bounds(self.model)

        # Crescimento máximo (λ_max) - Equação (2)
        solution = self.model.optimize()
        self.lmax = solution.objective_value if solution.objective_value is not None else 0.0
        print(f"Crescimento máximo (lmax): {self.lmax:.6f}")

        self._calculate_fva_c1()
        self._classify_reactions_c1()
        self._calculate_erog_c1_correct()         # ← definição corrigida
        self._calculate_chokepoints_c1_correct()
        self._validate_results()

        return True

    def _apply_model_bounds(self, model):
        """Condições padrão para iML1515 aeróbico em glicose (valores corrigidos 2026)"""
        print("Aplicando bounds corretos para iML1515 (aeróbico, glicose)...")

        # Resetar todas EX para conservador
        for rxn in model.reactions:
            if rxn.id.startswith("EX_"):
                rxn.bounds = (0, 1000)

        # Liberar saída de subprodutos comuns (boa prática)
        for ex in ["EX_co2_e", "EX_h2o_e", "EX_h_e", "EX_ac_e"]:
            try:
                model.reactions.get_by_id(ex).bounds = (-1000, 1000)
            except:
                pass

        # Valores corrigidos e recomendados
        ATPM_STANDARD_LOWER_BOUND = 8.39          # ← CORRETO para iML1515 (Monk et al. 2017)
        GLUCOSE_UPTAKE = -10.0                    # Conservador (muito usado); -18.5 para max growth
        OXYGEN_UPTAKE = -18.5                     # Valor padrão para E. coli iML1515

        try:
            model.reactions.EX_glc__D_e.lower_bound = GLUCOSE_UPTAKE
            model.reactions.EX_o2_e.lower_bound = OXYGEN_UPTAKE
            model.reactions.ATPM.lower_bound = ATPM_STANDARD_LOWER_BOUND

            biomass = model.reactions.get_by_id("BIOMASS_Ec_iML1515_core_75p37M")
            model.objective = biomass
            print(f"Biomassa: {biomass.id} | ATPM: {ATPM_STANDARD_LOWER_BOUND} | Glc: {GLUCOSE_UPTAKE} | O2: {OXYGEN_UPTAKE}")
        except KeyError as e:
            print(f"Reação essencial não encontrada: {e}. Verifique o modelo JSON!")
            # Adicionar verificação de IDs alternativos se necessário
            biomass_candidates = [r.id for r in model.reactions if 'BIOMASS' in r.id.upper()]
            print("Reações de biomassa encontradas:", biomass_candidates)
            glc_candidates = [r.id for r in model.reactions if 'glc' in r.id.lower() and 'EX' in r.id.upper()]
            print("Candidatos para glicose:", glc_candidates)

        # Nutrientes inorgânicos (sua lista está ótima, mantenha)
        inorganic = [
            "EX_nh4_e", "EX_pi_e", "EX_so4_e", "EX_mg2_e", "EX_k_e", "EX_fe2_e",
            "EX_ca2_e", "EX_cl_e", "EX_na1_e", "EX_zn2_e", "EX_mn2_e", "EX_cu2_e",
            "EX_ni2_e", "EX_mobd_e", "EX_cobalt2_e"
        ]
        for rid in inorganic:
            try:
                model.reactions.get_by_id(rid).lower_bound = -1000
            except:
                pass

    def _calculate_fva_c1(self):
        print("\nCalculando FVA para c=1 (crescimento ótimo)...")
        try:
            self.fva_result_c1 = flux_variability_analysis(
                self.model,
                fraction_of_optimum=1.0,
                processes=1
            )
            print("FVA c=1 concluído")
        except Exception as e:
            print(f"Erro FVA: {e}. Usando bounds originais como fallback.")
            self.fva_result_c1 = None

    def _classify_reactions_c1(self):
        print("\nClassificando reações (DR₁, RR₁, NR₁)...")
        self.dr_c1.clear()
        self.rr_c1.clear()
        self.nr_c1.clear()

        if self.fva_result_c1 is not None:
            for rid in self.fva_result_c1.index:
                min_f = self.fva_result_c1.loc[rid, 'minimum']
                max_f = self.fva_result_c1.loc[rid, 'maximum']

                if abs(min_f) < self.CONST_EPSILON and abs(max_f) < self.CONST_EPSILON:
                    self.dr_c1.add(rid)
                elif min_f < -self.CONST_EPSILON and max_f > self.CONST_EPSILON:
                    self.rr_c1.add(rid)
                else:
                    self.nr_c1.add(rid)
        else:
            # fallback
            for rxn in self.model.reactions:
                lb, ub = rxn.lower_bound, rxn.upper_bound
                if abs(lb) < self.CONST_EPSILON and abs(ub) < self.CONST_EPSILON:
                    self.dr_c1.add(rxn.id)
                elif lb < -self.CONST_EPSILON and ub > self.CONST_EPSILON:
                    self.rr_c1.add(rxn.id)
                else:
                    self.nr_c1.add(rxn.id)

        print(f"DR₁: {len(self.dr_c1)} | RR₁: {len(self.rr_c1)} | NR₁: {len(self.nr_c1)}")

    def _calculate_erog_c1_correct(self):
        """
        Cálculo FIDELIDADE ao artigo:
        Reação é EROG (ER₁) se growth após KO < lmax  (ou inviável)
        NENHUM threshold percentual artificial é aplicado.
        """
        print("\nCalculando EROG (ER₁) FIDELIDADE ao artigo CONTRABASS...")
        print(f"Referência λ_max = {self.lmax:.6f}")

        total = len(self.model.reactions)
        erog_count = 0
        essential_count = 0

        for i, rxn in enumerate(self.model.reactions, 1):
            model_ko = self.model.copy()
            ko_rxn = model_ko.reactions.get_by_id(rxn.id)
            ko_rxn.bounds = (0, 0)

            try:
                sol = model_ko.optimize()
                growth_ko = sol.objective_value if sol.objective_value is not None else 0.0
                if isnan(growth_ko):
                    growth_ko = 0.0
            except:
                growth_ko = 0.0

            is_essential = (growth_ko < self.CONST_EPSILON)
            is_erog = (growth_ko < self.lmax - self.CONST_EPSILON)  # ← definição correta

            # RGR sempre calculada (sua métrica excelente)
            rgr = max(0.0, (self.lmax - growth_ko) / self.lmax) if self.lmax > 1e-8 else 0.0

            # Pegar bounds FVA c=1
            flux_min = flux_max = 0.0
            if self.fva_result_c1 is not None and rxn.id in self.fva_result_c1.index:
                flux_min = self.fva_result_c1.loc[rxn.id, 'minimum']
                flux_max = self.fva_result_c1.loc[rxn.id, 'maximum']

            reaction_class = 'DR₁' if rxn.id in self.dr_c1 else \
                            'RR₁' if rxn.id in self.rr_c1 else \
                            'NR₁' if rxn.id in self.nr_c1 else 'UNKNOWN'

            self.erog_results[rxn.id] = {
                'is_erog': is_erog,
                'rgr': round(rgr, 6),
                'growth_ko': round(growth_ko, 6),
                'is_essential': is_essential,
                'lmax': self.lmax,
                'flux_min_c1': round(flux_min, 6),
                'flux_max_c1': round(flux_max, 6),
                'reaction_class_c1': reaction_class
            }

            if is_erog:
                erog_count += 1
            if is_essential:
                essential_count += 1

            if i % 200 == 0:
                print(f"  → {i}/{total} reações processadas")

        print(f"\nResumo EROG (fiel ao artigo):")
        print(f"• Reações EROG (growth_ko < λ_max): {erog_count}")
        print(f"• Reações essenciais (growth ≈ 0):   {essential_count}")

    def _calculate_chokepoints_c1_correct(self):
        """
        Calcula chokepoints em c=1 CORRETAMENTE conforme artigo CONTRABASS.
        
        Artigo Definição 2.5 (adaptada para c=1):
        Uma reação r é chokepoint se existe metabólito m tal que:
        m? = {r} (único consumidor) OU ?m = {r} (único produtor)
        
        Onde ?m = conjunto de produtores flux-dependent de m
        m? = conjunto de consumidores flux-dependent de m
        
        Considera direcionalidade com base em flux bounds de FVA c=1.
        """
        print("\nCalculando chokepoints para c=1 conforme artigo...")
        
        # Primeiro calcular conjuntos flux-dependent para c=1
        # Conforme Definições flux-dependent do artigo (Seção 2.4)
        
        chokepoints = set()
        metabolite_chokepoint_info = []
        
        for metabolite in self.model.metabolites:
            # Conjuntos para este metabólito
            producers = set() # ?m = conjunto de produtores
            consumers = set() # m? = conjunto de consumidores
            
            for reaction in metabolite.reactions:
                rid = reaction.id
                coeff = reaction.metabolites.get(metabolite, 0)
                
                if coeff == 0:
                    continue
                
                # Determinar flux bounds para c=1
                if self.fva_result_c1 is not None and rid in self.fva_result_c1.index:
                    flux_min = self.fva_result_c1.loc[rid, 'minimum']
                    flux_max = self.fva_result_c1.loc[rid, 'maximum']
                    
                    # Verificar se reação pode carregar fluxo em c=1
                    can_carry_flux = (abs(flux_max) > self.CONST_EPSILON or
                                      abs(flux_min) > self.CONST_EPSILON)
                    
                    if not can_carry_flux:
                        continue # Reação morta em c=1, não conta
                    
                    # Determinar se é produtor ou consumidor
                    # Baseado nas Definições flux-dependent do artigo:
                    # • m é produto de r se: (S(m,r) > 0 AND U[r] > 0) OR (S(m,r) < 0 AND L[r] < 0)
                    # • m é reagente de r se: (S(m,r) < 0 AND U[r] > 0) OR (S(m,r) > 0 AND L[r] < 0)
                    
                    # Para coeficiente positivo (m é produto quando fluxo positivo)
                    if coeff > 0:
                        # Produtor quando fluxo é positivo
                        if flux_max > self.CONST_EPSILON:
                            producers.add(rid)
                        # Consumidor quando reação reversa (fluxo negativo)
                        if flux_min < -self.CONST_EPSILON:
                            consumers.add(rid)
                    
                    # Para coeficiente negativo (m é reagente quando fluxo positivo)
                    elif coeff < 0:
                        # Consumidor quando fluxo é positivo
                        if flux_max > self.CONST_EPSILON:
                            consumers.add(rid)
                        # Produtor quando reação reversa (fluxo negativo)
                        if flux_min < -self.CONST_EPSILON:
                            producers.add(rid)
                
                else:
                    # Fallback: usar bounds originais
                    lb = reaction.lower_bound
                    ub = reaction.upper_bound
                    
                    if coeff > 0:
                        if ub > self.CONST_EPSILON:
                            producers.add(rid)
                        if lb < -self.CONST_EPSILON:
                            consumers.add(rid)
                    elif coeff < 0:
                        if ub > self.CONST_EPSILON:
                            consumers.add(rid)
                        if lb < -self.CONST_EPSILON:
                            producers.add(rid)
            
            # Verificar chokepoints conforme artigo
            # Reação é chokepoint se é único produtor OU único consumidor
            if len(producers) == 1:
                choke_id = list(producers)[0]
                chokepoints.add(choke_id)
                metabolite_chokepoint_info.append({
                    'metabolite': metabolite.id,
                    'reaction': choke_id,
                    'type': 'unique_producer',
                    'metabolite_name': metabolite.name
                })
            
            if len(consumers) == 1:
                choke_id = list(consumers)[0]
                chokepoints.add(choke_id)
                metabolite_chokepoint_info.append({
                    'metabolite': metabolite.id,
                    'reaction': choke_id,
                    'type': 'unique_consumer',
                    'metabolite_name': metabolite.name
                })
        
        self.chokepoints_c1 = chokepoints
        self.chokepoint_details = metabolite_chokepoint_info
        
        print(f" Total de chokepoints em c=1: {len(chokepoints)}")
        
        # Salvar detalhes dos chokepoints
        if metabolite_chokepoint_info:
            df_details = pd.DataFrame(metabolite_chokepoint_info)
            df_details.to_csv("CONTRABASS_c1_chokepoint_details.csv", index=False)
            print(f" Detalhes salvos em: CONTRABASS_c1_chokepoint_details.csv")

    def _validate_results(self):
        """Validação dos resultados conforme artigo."""
        print("\nValidando resultados CONTRABASS c=1...")
        
        # Estatísticas básicas
        total_erog = sum(1 for v in self.erog_results.values() if v['is_erog'])
        total_essential = sum(1 for v in self.erog_results.values() if v['is_essential'])
        
        # Distribuição de RGR
        rgr_distribution = {
            '0.0': 0, '0.0-0.1': 0, '0.1-0.5': 0,
            '0.5-0.9': 0, '0.9-1.0': 0, '1.0': 0
        }
        
        for data in self.erog_results.values():
            rgr = data['rgr']
            
            if rgr == 0.0:
                rgr_distribution['0.0'] += 1
            elif rgr < 0.1:
                rgr_distribution['0.0-0.1'] += 1
            elif rgr < 0.5:
                rgr_distribution['0.1-0.5'] += 1
            elif rgr < 0.9:
                rgr_distribution['0.5-0.9'] += 1
            elif rgr < 1.0:
                rgr_distribution['0.9-1.0'] += 1
            else: # rgr == 1.0
                rgr_distribution['1.0'] += 1
        
        # Contar interseções
        erog_and_chokepoint = sum(1 for rid in self.erog_results
                                  if self.erog_results[rid]['is_erog'] and rid in self.chokepoints_c1)
        
        # Análise por classificação de reação
        erog_by_class = {
            'DR₁': 0,
            'RR₁': 0,
            'NR₁': 0
        }
        
        choke_by_class = {
            'DR₁': 0,
            'RR₁': 0,
            'NR₁': 0
        }
        
        for rid, data in self.erog_results.items():
            reaction_class = data.get('reaction_class_c1', 'UNKNOWN')
            if data['is_erog'] and reaction_class in erog_by_class:
                erog_by_class[reaction_class] += 1
            
            if rid in self.chokepoints_c1 and reaction_class in choke_by_class:
                choke_by_class[reaction_class] += 1
        
        # Salvar estatísticas
        self.validation_stats = {
            'total_reactions': len(self.erog_results),
            'lmax': self.lmax,
            'erog_count': total_erog,
            'essential_count': total_essential,
            'chokepoints_c1': len(self.chokepoints_c1),
            'erog_and_chokepoint': erog_and_chokepoint,
            'dr_c1_count': len(self.dr_c1),
            'rr_c1_count': len(self.rr_c1),
            'nr_c1_count': len(self.nr_c1),
            'erog_by_class': erog_by_class,
            'choke_by_class': choke_by_class,
            'rgr_distribution': rgr_distribution
        }
        
        print(f" Reações EROG identificadas: {total_erog}")
        print(f" Reações essenciais (crescimento = 0): {total_essential}")
        print(f" Chokepoints em c=1: {len(self.chokepoints_c1)}")
        print(f" Reações EROG que também são chokepoints: {erog_and_chokepoint}")
        print(f"\n Classificação de reações em c=1:")
        print(f" Dead reactions (DR₁): {len(self.dr_c1)}")
        print(f" Reversible reactions (RR₁): {len(self.rr_c1)}")
        print(f" Non-reversible reactions (NR₁): {len(self.nr_c1)}")
        
        print(f"\n Distribuição de EROG por classe:")
        for class_type, count in erog_by_class.items():
            if len(self.erog_results) > 0:
                percentage = (count / total_erog) * 100 if total_erog > 0 else 0
                print(f" {class_type}: {count} reações ({percentage:.1f}% das EROG)")
        
        print(f"\n Distribuição de chokepoints por classe:")
        for class_type, count in choke_by_class.items():
            if len(self.chokepoints_c1) > 0:
                percentage = (count / len(self.chokepoints_c1)) * 100 if len(self.chokepoints_c1) > 0 else 0
                print(f" {class_type}: {count} reações ({percentage:.1f}% dos chokepoints)")
        
        print("\n Distribuição de RGR:")
        for category in ['0.0', '0.0-0.1', '0.1-0.5', '0.5-0.9', '0.9-1.0', '1.0']:
            count = rgr_distribution[category]
            percentage = (count / len(self.erog_results)) * 100
            print(f" {category}: {count} reações ({percentage:.1f}%)")

    def integrate_erog_data(self) -> pd.DataFrame:
        """
        Integra dados EROG/RGR com dataset existente.
        Segue estritamente metodologia CONTRABASS c=1.
        """
        print("\nIntegrando dados CONTRABASS c=1...")
        
        # Criar cópia do dataframe
        integrated_df = self.existing_df.copy()
        
        # Preparar listas para novas colunas
        erog_binary_values = []
        rgr_values = []
        growth_ko_values = []
        chokepoint_values = []
        flux_min_values = []
        flux_max_values = []
        is_essential_values = []
        reaction_class_values = []
        nr_c1_values = [] # Coluna específica para NR₁
        dr_c1_values = [] # Coluna específica para DR₁
        rr_c1_values = [] # Coluna específica para RR₁
        
        missing_reactions = []
        found_reactions = []
        
        for idx, row in integrated_df.iterrows():
            reaction_id = row['label']
            
            if reaction_id in self.erog_results:
                data = self.erog_results[reaction_id]
                
                # EROG binário (conforme artigo)
                erog_binary_values.append(1 if data['is_erog'] else 0)
                
                # RGR (nossa métrica de impacto)
                rgr_values.append(data['rgr'])
                
                # Crescimento após knockout
                growth_ko_values.append(data['growth_ko'])
                
                # Chokepoint em c=1
                chokepoint_values.append(1 if reaction_id in self.chokepoints_c1 else 0)
                
                # Flux bounds de FVA c=1
                flux_min_values.append(data.get('flux_min_c1', 0.0))
                flux_max_values.append(data.get('flux_max_c1', 0.0))
                
                # É essencial?
                is_essential_values.append(1 if data['is_essential'] else 0)
                
                # Classificação da reação
                reaction_class_values.append(data.get('reaction_class_c1', 'UNKNOWN'))
                
                # Colunas específicas para cada classe
                nr_c1_values.append(1 if reaction_id in self.nr_c1 else 0)
                dr_c1_values.append(1 if reaction_id in self.dr_c1 else 0)
                rr_c1_values.append(1 if reaction_id in self.rr_c1 else 0)
                
                found_reactions.append(reaction_id)
                
            else:
                # Reação não encontrada no modelo
                erog_binary_values.append(0)
                rgr_values.append(0.0)
                growth_ko_values.append(self.lmax) # Assume crescimento ótimo
                chokepoint_values.append(0)
                flux_min_values.append(0.0)
                flux_max_values.append(0.0)
                is_essential_values.append(0)
                reaction_class_values.append('NOT_FOUND')
                nr_c1_values.append(0)
                dr_c1_values.append(0)
                rr_c1_values.append(0)
                missing_reactions.append(reaction_id)
        
        # Adicionar novas colunas
        integrated_df['EROG_binary'] = erog_binary_values # Artigo CONTRABASS
        integrated_df['RGR'] = rgr_values # Nossa métrica
        integrated_df['growth_ko'] = growth_ko_values
        integrated_df['chokepoint_c1'] = chokepoint_values
        integrated_df['flux_min_c1'] = flux_min_values
        integrated_df['flux_max_c1'] = flux_max_values
        integrated_df['is_essential_c1'] = is_essential_values
        integrated_df['reaction_class_c1'] = reaction_class_values
        integrated_df['NR_c1'] = nr_c1_values # Non-reversible em c=1
        integrated_df['DR_c1'] = dr_c1_values # Dead reactions em c=1
        integrated_df['RR_c1'] = rr_c1_values # Reversible em c=1
        
        # Adicionar coluna de impacto qualitativo
        def categorize_impact(rgr, erog_binary):
            if erog_binary == 0:
                return "no_impact"
            elif rgr == 1.0:
                return "essential"
            elif rgr >= 0.5:
                return "high_impact"
            elif rgr >= 0.1:
                return "medium_impact"
            else:
                return "low_impact"
        
        integrated_df['impact_category'] = [
            categorize_impact(rgr, erog)
            for rgr, erog in zip(integrated_df['RGR'], integrated_df['EROG_binary'])
        ]
        
        # Estatísticas
        print(f"\n--- Estatísticas da Integração ---")
        print(f"Total de reações no dataset: {len(integrated_df)}")
        print(f"Reações encontradas no modelo: {len(found_reactions)}")
        print(f"Reações não encontradas: {len(missing_reactions)}")
        print(f"Reações EROG (binário): {integrated_df['EROG_binary'].sum()}")
        print(f"Reações chokepoint c=1: {integrated_df['chokepoint_c1'].sum()}")
        print(f"Reações essenciais (c=1): {integrated_df['is_essential_c1'].sum()}")
        print(f"Reações NR₁: {integrated_df['NR_c1'].sum()}")
        print(f"Reações DR₁: {integrated_df['DR_c1'].sum()}")
        print(f"Reações RR₁: {integrated_df['RR_c1'].sum()}")
        
        # Análise detalhada
        print("\n--- Análise Detalhada CONTRABASS c=1 ---")
        
        # Relação entre EROG e essentiality experimental
        if 'essentiality' in integrated_df.columns:
            essential_and_erog = ((integrated_df['essentiality'] == 1) &
                                  (integrated_df['EROG_binary'] == 1)).sum()
            essential_not_erog = ((integrated_df['essentiality'] == 1) &
                                  (integrated_df['EROG_binary'] == 0)).sum()
            erog_not_essential = ((integrated_df['essentiality'] == 0) &
                                  (integrated_df['EROG_binary'] == 1)).sum()
            
            print(f"Essencial experimental e EROG: {essential_and_erog}")
            print(f"Essencial experimental mas não EROG: {essential_not_erog}")
            print(f"EROG mas não essencial experimental: {erog_not_essential}")
        
        # Reações EROG que são chokepoints (alvos prioritários)
        erog_and_chokepoint = ((integrated_df['EROG_binary'] == 1) &
                              (integrated_df['chokepoint_c1'] == 1)).sum()
        print(f"EROG e chokepoint (alvos prioritários): {erog_and_chokepoint}")
        
        # Análise por classe de reação
        print(f"\nAnálise por classe de reação (c=1):")
        for class_name in ['NR_c1', 'DR_c1', 'RR_c1']:
            class_count = integrated_df[class_name].sum()
            class_erog = ((integrated_df[class_name] == 1) &
                         (integrated_df['EROG_binary'] == 1)).sum()
            class_choke = ((integrated_df[class_name] == 1) &
                          (integrated_df['chokepoint_c1'] == 1)).sum()
            
            if class_count > 0:
                erog_percentage = (class_erog / class_count) * 100
                choke_percentage = (class_choke / class_count) * 100
                print(f" {class_name}: {class_count} reações")
                print(f" - EROG: {class_erog} ({erog_percentage:.1f}%)")
                print(f" - Chokepoints: {class_choke} ({choke_percentage:.1f}%)")
        
        # Distribuição de categorias de impacto
        print("\nDistribuição de Impacto (RGR):")
        for category in ['no_impact', 'low_impact', 'medium_impact', 'high_impact', 'essential']:
            count = (integrated_df['impact_category'] == category).sum()
            percentage = (count / len(integrated_df)) * 100
            print(f" {category}: {count} reações ({percentage:.1f}%)")
        
        return integrated_df

    def generate_detailed_report(self, integrated_df: pd.DataFrame, output_path: str = "CONTRABASS_c1_detailed_report.txt"):
        """Gera relatório detalhado da análise CONTRABASS c=1."""
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("RELATÓRIO DETALHADO - CONTRABASS c=1 (EROG) com Extensão RGR")
        report_lines.append("Implementação conforme: Oarga et al. Bioinformatics, 39(2), 2023")
        report_lines.append("="*80)
        
        report_lines.append(f"\n1. INFORMAÇÕES DO MODELO")
        report_lines.append(f" Modelo: {self.model_path}")
        report_lines.append(f" Reações no modelo: {len(self.model.reactions)}")
        report_lines.append(f" Metabólitos no modelo: {len(self.model.metabolites)}")
        report_lines.append(f" Crescimento máximo (lmax): {self.lmax:.6f}")
        
        report_lines.append(f"\n2. RESULTADOS CONTRABASS c=1 (CONFORME ARTIGO)")
        report_lines.append(f" Reações EROG identificadas (ER₁): {self.validation_stats['erog_count']}")
        report_lines.append(f" Reações essenciais (crescimento = 0): {self.validation_stats['essential_count']}")
        report_lines.append(f" Chokepoints em c=1 (CP₁): {len(self.chokepoints_c1)}")
        report_lines.append(f" Non-reversible reactions em c=1 (NR₁): {len(self.nr_c1)}")
        report_lines.append(f" Dead reactions em c=1 (DR₁): {len(self.dr_c1)}")
        report_lines.append(f" Reversible reactions em c=1 (RR₁): {len(self.rr_c1)}")
        report_lines.append(f" Tolerância numérica (ε): {self.CONST_EPSILON}")
        
        report_lines.append(f"\n3. INTERPRETAÇÃO DAS MÉTRICAS CONFORME ARTIGO")
        report_lines.append(f" EROG_binary = 1: Knockout reduz crescimento abaixo de lmax (artigo CONTRABASS)")
        report_lines.append(f" RGR: Redução proporcional no crescimento (0=nada, 1=essencial) - NOSSA EXTENSÃO")
        report_lines.append(f" chokepoint_c1 = 1: Único produtor/consumidor em crescimento ótimo (Def. 2.5)")
        report_lines.append(f" is_essential_c1 = 1: Knockout resulta em crescimento = 0 (Def. 2.2)")
        report_lines.append(f" NR_c1 = 1: Reação não-reversível em c=1 (Def. 2.9)")
        report_lines.append(f" DR_c1 = 1: Reação morta em c=1 (fluxo = 0) (Def. 2.7)")
        report_lines.append(f" RR_c1 = 1: Reação reversível em c=1 (Def. 2.8)")
        
        report_lines.append(f"\n4. TOP 20 REAÇÕES EROG POR IMPACTO (RGR)")
        # Top 20 reações EROG por RGR
        if 'RGR' in integrated_df.columns and 'EROG_binary' in integrated_df.columns:
            top_erog = integrated_df[integrated_df['EROG_binary'] == 1].nlargest(20, 'RGR')
            for idx, row in top_erog.iterrows():
                flags = []
                if 'essentiality' in row and row['essentiality'] == 1:
                    flags.append("ESS_exp")
                if row['chokepoint_c1'] == 1:
                    flags.append("CHOKE")
                if row['is_essential_c1'] == 1:
                    flags.append("ESS_c1")
                if row['NR_c1'] == 1:
                    flags.append("NR₁")
                if row['DR_c1'] == 1:
                    flags.append("DR₁")
                if row['RR_c1'] == 1:
                    flags.append("RR₁")
                
                flag_str = f" [{', '.join(flags)}]" if flags else ""
                report_lines.append(f" {row['label']}: RGR={row['RGR']:.4f}, "
                                  f"growth_ko={row['growth_ko']:.4f}{flag_str}")
        else:
            report_lines.append(" Dados insuficientes para gerar top EROG")
        
        report_lines.append(f"\n5. TOP 10 ALVOS PRIORITÁRIOS (EROG + CHOKEPOINT)")
        if all(col in integrated_df.columns for col in ['EROG_binary', 'chokepoint_c1', 'RGR']):
            priority_targets = integrated_df[(integrated_df['EROG_binary'] == 1) &
                                            (integrated_df['chokepoint_c1'] == 1)]
            priority_targets = priority_targets.nlargest(10, 'RGR')
            
            for idx, row in priority_targets.iterrows():
                exp_ess = " (exp_essential)" if 'essentiality' in row and row['essentiality'] == 1 else ""
                class_info = f" [{row.get('reaction_class_c1', '')}]" if 'reaction_class_c1' in row else ""
                report_lines.append(f" {row['label']}: RGR={row['RGR']:.4f}{class_info}{exp_ess}")
        else:
            report_lines.append(" Dados insuficientes para identificar alvos prioritários")
        
        report_lines.append(f"\n6. ANÁLISE DE CLASSES DE REAÇÃO (c=1)")
        if 'reaction_class_c1' in integrated_df.columns:
            class_counts = integrated_df['reaction_class_c1'].value_counts()
            for class_name, count in class_counts.items():
                erog_in_class = ((integrated_df['reaction_class_c1'] == class_name) &
                                (integrated_df['EROG_binary'] == 1)).sum()
                choke_in_class = ((integrated_df['reaction_class_c1'] == class_name) &
                                 (integrated_df['chokepoint_c1'] == 1)).sum()
                
                erog_percentage = (erog_in_class / count * 100) if count > 0 else 0
                choke_percentage = (choke_in_class / count * 100) if count > 0 else 0
                
                report_lines.append(f" {class_name}: {count} reações")
                report_lines.append(f" • EROG: {erog_in_class} ({erog_percentage:.1f}%)")
                report_lines.append(f" • Chokepoints: {choke_in_class} ({choke_percentage:.1f}%)")
        
        report_lines.append(f"\n7. METODOLOGIA CONTRABASS APLICADA")
        report_lines.append(" • c=1: Crescimento ótimo (lmax)")
        report_lines.append(" • EROG: ER₁ = reações essenciais para crescimento ótimo (Def. 2.10)")
        report_lines.append(" • Definição: Reação r é EROG se knockout resulta em crescimento < lmax")
        report_lines.append(" • Chokepoints c=1: Baseados em flux bounds de FVA com c=1 (Def. 2.5)")
        report_lines.append(" • NR₁: Reações não-reversíveis em c=1 (Def. 2.9)")
        report_lines.append(" • DR₁: Reações mortas em c=1 (fluxo = 0) (Def. 2.7)")
        report_lines.append(" • RR₁: Reações reversíveis em c=1 (Def. 2.8)")
        report_lines.append(" • RGR: Métrica adicional (0-1) que quantifica impacto no crescimento")
        
        report_lines.append(f"\n8. VALIDAÇÃO")
        report_lines.append(f" Total de reações processadas: {self.validation_stats['total_reactions']}")
        report_lines.append(f" Reações EROG e chokepoint: {self.validation_stats['erog_and_chokepoint']}")
        report_lines.append(f" EROG por classe: NR₁={self.validation_stats['erog_by_class'].get('NR₁', 0)}, "
                          f"RR₁={self.validation_stats['erog_by_class'].get('RR₁', 0)}, "
                          f"DR₁={self.validation_stats['erog_by_class'].get('DR₁', 0)}")
        
        report_lines.append("\n" + "="*80)
        
        # Salvar relatório
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\nRelatório detalhado salvo em: {output_path}")
        
        # Imprimir resumo no console
        print('\n'.join(report_lines[:50])) # Imprime apenas as primeiras linhas no console

    def save_additional_results(self, integrated_df: pd.DataFrame):
        """Salva resultados adicionais em arquivos Excel separados."""
        
        # 1. Salvar todas as reações EROG
        erog_df = integrated_df[integrated_df['EROG_binary'] == 1].copy()
        erog_df = erog_df.sort_values('RGR', ascending=False)
        erog_df.to_excel("CONTRABASS_c1_all_EROG_reactions.xlsx", index=False)
        print(f" Todas as reações EROG salvas em: CONTRABASS_c1_all_EROG_reactions.xlsx")
        
        # 2. Salvar reações prioritárias (EROG + chokepoint)
        priority_df = integrated_df[(integrated_df['EROG_binary'] == 1) &
                                   (integrated_df['chokepoint_c1'] == 1)].copy()
        priority_df = priority_df.sort_values('RGR', ascending=False)
        priority_df.to_excel("CONTRABASS_c1_priority_targets.xlsx", index=False)
        print(f" Alvos prioritários salvos em: CONTRABASS_c1_priority_targets.xlsx")
        
        # 3. Salvar reações essenciais (c=1)
        essential_df = integrated_df[integrated_df['is_essential_c1'] == 1].copy()
        essential_df.to_excel("CONTRABASS_c1_essential_reactions.xlsx", index=False)
        print(f" Reações essenciais (c=1) salvas em: CONTRABASS_c1_essential_reactions.xlsx")
        
        # 4. Salvar reações NR₁ (non-reversible em c=1)
        nr_c1_df = integrated_df[integrated_df['NR_c1'] == 1].copy()
        nr_c1_df.to_excel("CONTRABASS_c1_NR_reactions.xlsx", index=False)
        print(f" Reações NR₁ salvas em: CONTRABASS_c1_NR_reactions.xlsx")
        
        # 5. Salvar reações DR₁ (dead reactions em c=1)
        dr_c1_df = integrated_df[integrated_df['DR_c1'] == 1].copy()
        dr_c1_df.to_excel("CONTRABASS_c1_DR_reactions.xlsx", index=False)
        print(f" Reações DR₁ salvas em: CONTRABASS_c1_DR_reactions.xlsx")
        
        # 6. Salvar estatísticas resumidas
        stats_df = pd.DataFrame([self.validation_stats])
        stats_df.to_excel("CONTRABASS_c1_summary_statistics.xlsx", index=False)
        print(f" Estatísticas resumidas salvas em: CONTRABASS_c1_summary_statistics.xlsx")
        
        # 7. Salvar lista de chokepoints
        choke_list = pd.DataFrame(list(self.chokepoints_c1), columns=['reaction_id'])
        choke_list.to_excel("CONTRABASS_c1_chokepoints_list.xlsx", index=False)
        print(f" Lista de chokepoints salva em: CONTRABASS_c1_chokepoints_list.xlsx")
        
        # 8. Salvar detalhes completos de classificação
        classification_details = []
        for rid, data in self.erog_results.items():
            classification_details.append({
                'reaction_id': rid,
                'is_erog': data['is_erog'],
                'rgr': data['rgr'],
                'growth_ko': data['growth_ko'],
                'is_essential': data['is_essential'],
                'flux_min_c1': data.get('flux_min_c1', 0.0),
                'flux_max_c1': data.get('flux_max_c1', 0.0),
                'reaction_class_c1': data.get('reaction_class_c1', 'UNKNOWN'),
                'in_chokepoints_c1': 1 if rid in self.chokepoints_c1 else 0,
                'in_nr_c1': 1 if rid in self.nr_c1 else 0,
                'in_dr_c1': 1 if rid in self.dr_c1 else 0,
                'in_rr_c1': 1 if rid in self.rr_c1 else 0
            })
        
        class_df = pd.DataFrame(classification_details)
        class_df.to_excel("CONTRABASS_c1_complete_classification.xlsx", index=False)
        print(f" Classificação completa salva em: CONTRABASS_c1_complete_classification.xlsx")

def main():
    """Função principal para executar a análise CONTRABASS c=1."""
    
    # Configurações
    MODEL_PATH = "iML1515_glucose.json"
    EXISTING_CSV_PATH = "iML1515_mfg_nodes_ess-label_fba_pred.csv"
    OUTPUT_PATH = "iML1515_CONTRABASS_c1_results.csv"
    REPORT_PATH = "CONTRABASS_c1_detailed_report.txt"
    
    try:
        print("="*80)
        print("CONTRABASS c=1 - Análise de Vulnerabilidades Metabólicas")
        print("Implementação CORRIGIDA conforme: Oarga et al. Bioinformatics, 39(2), 2023")
        print("Foco: EROG (ER₁) - Essential Reactions for Optimal Growth (c=1)")
        print("Inclui: NR₁, DR₁, RR₁, CP₁ conforme definições do artigo")
        print("="*80)
        
        # Inicializar analisador
        analyzer = CONTRABASS_EROG(MODEL_PATH, EXISTING_CSV_PATH)
        
        # Carregar e validar dados
        print("\n[1/4] Carregando dados e executando CONTRABASS c=1...")
        analyzer.load_and_validate_data()
        
        # Integrar dados
        print("\n[2/4] Integrando resultados CONTRABASS...")
        integrated_df = analyzer.integrate_erog_data()
        
        # Salvar resultados principais
        integrated_df.to_csv(OUTPUT_PATH, index=False)
        print(f"\nResultados principais salvos em: {OUTPUT_PATH}")
        
        # Gerar relatório
        print("\n[3/4] Gerando relatório analítico CONTRABASS...")
        analyzer.generate_detailed_report(integrated_df, REPORT_PATH)
        
        # Salvar resultados adicionais
        print("\n[4/4] Salvando resultados adicionais...")
        analyzer.save_additional_results(integrated_df)
        
        print("\n" + "="*80)
        print("ANÁLISE CONTRABASS c=1 CONCLUÍDA COM SUCESSO!")
        print("="*80)
        
        # Instruções finais
        print("\n📊 ARQUIVOS GERADOS:")
        print("1. iML1515_CONTRABASS_c1_results.csv - Dataset completo com todas as colunas")
        print("2. CONTRABASS_c1_all_EROG_reactions.xlsx - Todas as reações EROG (ordenadas por RGR)")
        print("3. CONTRABASS_c1_priority_targets.xlsx - Alvos prioritários (EROG + chokepoint)")
        print("4. CONTRABASS_c1_essential_reactions.xlsx - Reações essenciais (crescimento = 0)")
        print("5. CONTRABASS_c1_NR_reactions.xlsx - Reações não-reversíveis em c=1 (NR₁)")
        print("6. CONTRABASS_c1_DR_reactions.xlsx - Reações mortas em c=1 (DR₁)")
        print("7. CONTRABASS_c1_summary_statistics.xlsx - Estatísticas resumidas")
        print("8. CONTRABASS_c1_chokepoints_list.xlsx - Lista de chokepoints c=1")
        print("9. CONTRABASS_c1_chokepoint_details.csv - Detalhes dos chokepoints")
        print("10. CONTRABASS_c1_complete_classification.xlsx - Classificação completa")
        print("11. CONTRABASS_c1_detailed_report.txt - Relatório detalhado")
        
        print("\n🎯 INTERPRETAÇÃO DOS RESULTADOS (conforme artigo):")
        print("• EROG_binary = 1: Deleção impede crescimento ÓTIMO (lmax) - Def. 2.10")
        print("• NR_c1 = 1: Reação não-reversível em c=1 - Def. 2.9")
        print("• DR_c1 = 1: Reação morta em c=1 - Def. 2.7")
        print("• RR_c1 = 1: Reação reversível em c=1 - Def. 2.8")
        print("• chokepoint_c1 = 1: Único produtor/consumidor em c=1 - Def. 2.5")
        print("• RGR: Métrica adicional (0-1) que quantifica impacto no crescimento")
        print("• Alvos ideais: EROG_binary=1 E chokepoint_c1=1 E RGR alto")
        
    except Exception as e:
        print(f"\n❌ ERRO na análise CONTRABASS: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()