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
    Adaptado para Plasmodium falciparum iAM-Pf480.
    
    Definições CORRETAS do artigo:
    - EROG = ER₁ = Essential Reactions for Optimal Growth (c=1)
    - Uma reação é EROG se knockout resulta em crescimento < lmax 
      (ou inviável) com restrição c=1
    
    NOTA SOBRE DISCREPÂNCIA:
    O artigo original reporta 317 EROG para iAM-Pf480.
    Este código usa threshold relativo de 0.01% do lmax para evitar
    falsos positivos por imprecisão numérica.
    """

    def __init__(self, model_path: str, existing_csv_path: str):
        self.model_path = model_path
        self.existing_csv_path = existing_csv_path
        self.CONST_EPSILON = 1e-6           # tolerância numérica base
        self.RELATIVE_THRESHOLD = 0.0001    # 0.01% do lmax (threshold relativo)
        self.model = None
        self.lmax = 0.0
        self.effective_threshold = 0.0       # threshold efetivo usado
        self.biomass_reaction_id = None      # ID da reação de biomassa
        self.erog_results = {}              # {reaction_id: {...}}
        self.fva_result_c1 = None
        self.chokepoints_c1 = set()
        self.nr_c1 = set()
        self.dr_c1 = set()
        self.rr_c1 = set()
        self.validation_stats = {}
        self.erog_differences = []          # Para diagnóstico

    def _get_biomass_reaction_id(self) -> str:
        """
        Obtém o ID da reação de biomassa do modelo.
        Tenta vários métodos para lidar com diferentes versões do COBRApy.
        """
        # Método 1: Tentar pegar do objective
        try:
            if hasattr(self.model.objective, 'variable'):
                # COBRApy mais recente
                return self.model.objective.variable.name
            elif hasattr(self.model.objective, 'expression'):
                # COBRApy antigo
                expr_str = str(self.model.objective.expression)
                # Extrair o ID da reação da expressão
                # Exemplo: "1.0*BIOMASS__4 - 1.0*BIOMASS__4_reverse..."
                import re
                match = re.search(r'([A-Za-z0-9_]+)', expr_str)
                if match:
                    return match.group(1)
        except:
            pass
        
        # Método 2: Procurar reação com objective coefficient > 0
        for rxn in self.model.reactions:
            if rxn.objective_coefficient > 0:
                return rxn.id
        
        # Método 3: Fallback para IDs conhecidos
        for candidate in ['BIOMASS_Pf_iAM_Pf480', 'BIOMASS__4', 'BIOMASS']:
            try:
                self.model.reactions.get_by_id(candidate)
                return candidate
            except:
                continue
        
        return None

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
        
        # Obter ID da biomassa
        self.biomass_reaction_id = self._get_biomass_reaction_id()
        print(f"Reação de biomassa identificada: {self.biomass_reaction_id}")

        # Crescimento máximo (λ_max) - Equação (2)
        solution = self.model.optimize()
        self.lmax = solution.objective_value if solution.objective_value is not None else 0.0
        print(f"Crescimento máximo (lmax): {self.lmax:.6f}")
        
        # Calcular threshold efetivo
        self.effective_threshold = max(self.CONST_EPSILON, self.lmax * self.RELATIVE_THRESHOLD)
        print(f"Threshold efetivo: {self.effective_threshold:.8f}")
        print(f"Threshold relativo: {self.RELATIVE_THRESHOLD*100:.3f}% do lmax")

        self._calculate_fva_c1()
        self._classify_reactions_c1()
        self._calculate_erog_c1_correct()
        self._calculate_chokepoints_c1_correct()
        self._validate_results()
        self._diagnose_erog_discrepancy()

        return True

    def _apply_model_bounds(self, model):
        """
        Aplica as condições de contorno para o modelo iAM-Pf480.
        Como o modelo JSON já deve conter os bounds apropriados para
        o cultivo in vitro do parasita, apenas garantimos que a reação
        de biomassa seja o objetivo.
        """
        print("\nConfigurando modelo para P. falciparum iAM-Pf480...")

        # Definir a reação de biomassa como objetivo
        # Tentar vários IDs comuns de biomassa do iAM-Pf480
        biomass_candidates = [
            'BIOMASS_Pf_iAM_Pf480',
            'BIOMASS__4',
            'BIOMASS'
        ]
        
        biomass_found = False
        for biomass_id in biomass_candidates:
            try:
                biomass = model.reactions.get_by_id(biomass_id)
                model.objective = biomass
                print(f"Biomassa definida: {biomass.id}")
                print(f"Bounds da biomassa: [{biomass.lower_bound}, {biomass.upper_bound}]")
                biomass_found = True
                break
            except KeyError:
                continue
        
        if not biomass_found:
            print("Procurando reações de biomassa alternativas...")
            biomass_candidates_ids = [r.id for r in model.reactions if 'BIOMASS' in r.id.upper()]
            print("Candidatas encontradas:", biomass_candidates_ids)
            if biomass_candidates_ids:
                fallback_biomass_id = biomass_candidates_ids[0]
                print(f"Usando '{fallback_biomass_id}' como reação de biomassa (fallback).")
                model.objective = model.reactions.get_by_id(fallback_biomass_id)
            else:
                raise ValueError("Nenhuma reação de biomassa encontrada no modelo.")

        print("Usando bounds do arquivo JSON para trocas com o meio (EX_).")

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
        Cálculo FIDELIDADE ao artigo CONTRABASS.
        Reação é EROG (ER₁) se growth após KO < lmax - effective_threshold.
        
        O threshold efetivo é o máximo entre:
        - CONST_EPSILON (1e-6, tolerância numérica absoluta)
        - RELATIVE_THRESHOLD * lmax (0.01% do lmax, threshold relativo)
        
        Isso evita classificar como EROG reações cujo knockout causa
        redução insignificante no crescimento devido a imprecisão numérica.
        """
        print("\nCalculando EROG (ER₁) FIDELIDADE ao artigo CONTRABASS...")
        print(f"Referência λ_max = {self.lmax:.6f}")
        print(f"Threshold efetivo = {self.effective_threshold:.8f}")

        total = len(self.model.reactions)
        erog_count = 0
        essential_count = 0
        
        # Para diagnóstico: guardar diferenças para análise
        self.erog_differences = []

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

            # Threshold para essencial: crescimento menor que threshold
            is_essential = (growth_ko < self.effective_threshold)
            
            # Threshold para EROG: crescimento menor que lmax - threshold
            is_erog = (growth_ko < self.lmax - self.effective_threshold)

            # RGR (Relative Growth Reduction)
            rgr = max(0.0, (self.lmax - growth_ko) / self.lmax) if self.lmax > 1e-8 else 0.0

            # Guardar para diagnóstico
            if growth_ko < self.lmax:
                self.erog_differences.append({
                    'rid': rxn.id,
                    'growth_ko': growth_ko,
                    'diff_from_lmax': self.lmax - growth_ko,
                    'is_erog': is_erog,
                    'rgr': rgr
                })

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
                'reaction_class_c1': reaction_class,
                'diff_from_lmax': round(self.lmax - growth_ko, 8)
            }

            if is_erog:
                erog_count += 1
            if is_essential:
                essential_count += 1

            if i % 200 == 0:
                print(f"  → {i}/{total} reações processadas")

        print(f"\nResumo EROG (fiel ao artigo):")
        print(f"• Reações EROG (growth_ko < λ_max - ε_eff): {erog_count}")
        print(f"• Reações essenciais (growth ≈ 0): {essential_count}")
        print(f"• Artigo reporta: 317 EROG, 192 essenciais")
        print(f"• Diferença EROG: {erog_count - 317:+d}")
        print(f"• Diferença essenciais: {essential_count - 192:+d}")

    def _diagnose_erog_discrepancy(self):
        """
        Diagnostica a diferença entre os EROG encontrados e os 317 reportados no artigo.
        """
        print("\n" + "="*80)
        print("DIAGNÓSTICO DE DISCREPÂNCIA EROG")
        print(f"Artigo reporta 317 EROG, código encontra {self.validation_stats['erog_count']}")
        print("="*80)
        
        # Ordenar reações por diferença do lmax (menor diferença = mais próximas do lmax)
        self.erog_differences.sort(key=lambda x: x['diff_from_lmax'])
        
        current_erog = self.validation_stats['erog_count']
        target_erog = 317
        
        if current_erog == target_erog:
            print("\n✅ CONTAGEM DE EROG COINCIDE COM O ARTIGO!")
            return
        
        print(f"\n1. Distribuição de diferenças (lmax - growth_ko):")
        if len(self.erog_differences) > 0:
            diffs = [d['diff_from_lmax'] for d in self.erog_differences]
            print(f"   Mínimo: {min(diffs):.2e}")
            print(f"   Máximo: {max(diffs):.2e}")
            print(f"   Mediano: {np.median(diffs):.2e}")
        
        # Análise de diferentes thresholds
        print(f"\n2. Contagem de EROG com diferentes thresholds:")
        test_thresholds = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
        test_thresholds += [self.lmax * 0.01, self.lmax * 0.001, self.lmax * 0.0001, 
                           self.lmax * 0.00001, self.lmax * 0.000001, self.lmax * 0.00005]
        
        for threshold in sorted(set(test_thresholds)):
            count = sum(1 for d in self.erog_differences if d['diff_from_lmax'] > threshold)
            diff_from_article = count - target_erog
            if abs(diff_from_article) <= 20 or threshold in [1e-6, self.lmax * 0.0001, self.lmax * 0.00005]:
                print(f"   ε = {threshold:.2e} ({threshold/self.lmax*100:.4f}% lmax): "
                      f"{count} EROG (diferença: {diff_from_article:+d})")
        
        # Encontrar threshold ideal para 317 EROG
        if current_erog > target_erog:
            print(f"\n3. Para OBTER EXATAMENTE {target_erog} EROG:")
            excess = current_erog - target_erog
            print(f"   Excesso atual: {excess} reações")
            
            # Pegar as reações EROG com menor diferença do lmax (as mais incertas)
            erog_reactions = [d for d in self.erog_differences if d['is_erog']]
            erog_reactions.sort(key=lambda x: x['diff_from_lmax'])
            
            if excess <= len(erog_reactions):
                # O threshold seria a diferença da última reação a ser removida
                new_threshold = erog_reactions[excess - 1]['diff_from_lmax'] * 1.0001
                print(f"   Sugestão de threshold: ε = {new_threshold:.8f}")
                print(f"   Threshold relativo: {new_threshold/self.lmax*100:.4f}% do lmax")
                print(f"   Ajuste RELATIVE_THRESHOLD para: {new_threshold/self.lmax:.8f}")
                
                print(f"\n   Top {min(20, excess)} reações que seriam REMOVIDAS dos EROG:")
                for i, rxn in enumerate(erog_reactions[:min(20, excess)]):
                    print(f"   {i+1:2d}. {rxn['rid']:30s}: growth_ko={rxn['growth_ko']:.6f}, "
                          f"diff={rxn['diff_from_lmax']:.2e}, RGR={rxn['rgr']:.6f}")
        
        print(f"\n4. Informações do modelo:")
        print(f"   Reação de biomassa: {self.biomass_reaction_id}")
        print(f"   lmax atual: {self.lmax:.8f}")
        print(f"   Threshold efetivo atual: {self.effective_threshold:.8f}")
        print(f"   Threshold relativo atual: {self.RELATIVE_THRESHOLD*100:.4f}% do lmax")
        print(f"   Total de reações no modelo: {len(self.model.reactions)}")
        print(f"   Total de metabólitos no modelo: {len(self.model.metabolites)}")
        
        print(f"\n5. Comparação com artigo (Figura 3 do artigo):")
        print(f"   | Conjunto       | Artigo | Código | Diferença |")
        print(f"   |----------------|--------|--------|-----------|")
        print(f"   | ER (essencial) | 192    | {self.validation_stats['essential_count']:3d}     | {self.validation_stats['essential_count']-192:+3d}         |")
        print(f"   | EROG (ER₁)     | 317    | {self.validation_stats['erog_count']:3d}     | {self.validation_stats['erog_count']-317:+3d}         |")
        print(f"   | NR₁            | ~504   | {len(self.nr_c1):3d}     | -         |")
        print(f"   | DR₁            | ~469   | {len(self.dr_c1):3d}     | -         |")
        print(f"   | RR₁            | ~110   | {len(self.rr_c1):3d}     | -         |")
        print(f"   | CP₁            | ~416   | {len(self.chokepoints_c1):3d}     | -         |")
        print(f"   * Valores aproximados do artigo podem variar conforme condições")
        print("="*80)

    def _calculate_chokepoints_c1_correct(self):
        """
        Calcula chokepoints em c=1 CORRETAMENTE conforme artigo CONTRABASS.
        """
        print("\nCalculando chokepoints para c=1 conforme artigo...")
        
        chokepoints = set()
        metabolite_chokepoint_info = []
        
        for metabolite in self.model.metabolites:
            producers = set()
            consumers = set()
            
            for reaction in metabolite.reactions:
                rid = reaction.id
                coeff = reaction.metabolites.get(metabolite, 0)
                
                if coeff == 0:
                    continue
                
                if self.fva_result_c1 is not None and rid in self.fva_result_c1.index:
                    flux_min = self.fva_result_c1.loc[rid, 'minimum']
                    flux_max = self.fva_result_c1.loc[rid, 'maximum']
                    
                    can_carry_flux = (abs(flux_max) > self.CONST_EPSILON or
                                      abs(flux_min) > self.CONST_EPSILON)
                    
                    if not can_carry_flux:
                        continue
                    
                    if coeff > 0:
                        if flux_max > self.CONST_EPSILON:
                            producers.add(rid)
                        if flux_min < -self.CONST_EPSILON:
                            consumers.add(rid)
                    
                    elif coeff < 0:
                        if flux_max > self.CONST_EPSILON:
                            consumers.add(rid)
                        if flux_min < -self.CONST_EPSILON:
                            producers.add(rid)
                
                else:
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
        
        if metabolite_chokepoint_info:
            df_details = pd.DataFrame(metabolite_chokepoint_info)
            df_details.to_csv("Pf_CONTRABASS_c1_chokepoint_details.csv", index=False)
            print(f" Detalhes salvos em: Pf_CONTRABASS_c1_chokepoint_details.csv")

    def _validate_results(self):
        """Validação dos resultados conforme artigo."""
        print("\nValidando resultados CONTRABASS c=1...")
        
        total_erog = sum(1 for v in self.erog_results.values() if v['is_erog'])
        total_essential = sum(1 for v in self.erog_results.values() if v['is_essential'])
        
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
            else:
                rgr_distribution['1.0'] += 1
        
        erog_and_chokepoint = sum(1 for rid in self.erog_results
                                  if self.erog_results[rid]['is_erog'] and rid in self.chokepoints_c1)
        
        erog_by_class = {'DR₁': 0, 'RR₁': 0, 'NR₁': 0}
        choke_by_class = {'DR₁': 0, 'RR₁': 0, 'NR₁': 0}
        
        for rid, data in self.erog_results.items():
            reaction_class = data.get('reaction_class_c1', 'UNKNOWN')
            if data['is_erog'] and reaction_class in erog_by_class:
                erog_by_class[reaction_class] += 1
            
            if rid in self.chokepoints_c1 and reaction_class in choke_by_class:
                choke_by_class[reaction_class] += 1
        
        self.validation_stats = {
            'total_reactions': len(self.erog_results),
            'lmax': self.lmax,
            'effective_threshold': self.effective_threshold,
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

    def integrate_erog_data(self) -> pd.DataFrame:
        """Integra dados EROG/RGR com dataset existente."""
        print("\nIntegrando dados CONTRABASS c=1...")
        
        integrated_df = self.existing_df.copy()
        
        erog_binary_values = []
        rgr_values = []
        growth_ko_values = []
        chokepoint_values = []
        flux_min_values = []
        flux_max_values = []
        is_essential_values = []
        reaction_class_values = []
        nr_c1_values = []
        dr_c1_values = []
        rr_c1_values = []
        diff_from_lmax_values = []
        
        missing_reactions = []
        found_reactions = []
        
        for idx, row in integrated_df.iterrows():
            reaction_id = row['label']
            
            if reaction_id in self.erog_results:
                data = self.erog_results[reaction_id]
                
                erog_binary_values.append(1 if data['is_erog'] else 0)
                rgr_values.append(data['rgr'])
                growth_ko_values.append(data['growth_ko'])
                chokepoint_values.append(1 if reaction_id in self.chokepoints_c1 else 0)
                flux_min_values.append(data.get('flux_min_c1', 0.0))
                flux_max_values.append(data.get('flux_max_c1', 0.0))
                is_essential_values.append(1 if data['is_essential'] else 0)
                reaction_class_values.append(data.get('reaction_class_c1', 'UNKNOWN'))
                nr_c1_values.append(1 if reaction_id in self.nr_c1 else 0)
                dr_c1_values.append(1 if reaction_id in self.dr_c1 else 0)
                rr_c1_values.append(1 if reaction_id in self.rr_c1 else 0)
                diff_from_lmax_values.append(data.get('diff_from_lmax', 0.0))
                
                found_reactions.append(reaction_id)
            else:
                erog_binary_values.append(0)
                rgr_values.append(0.0)
                growth_ko_values.append(self.lmax)
                chokepoint_values.append(0)
                flux_min_values.append(0.0)
                flux_max_values.append(0.0)
                is_essential_values.append(0)
                reaction_class_values.append('NOT_FOUND')
                nr_c1_values.append(0)
                dr_c1_values.append(0)
                rr_c1_values.append(0)
                diff_from_lmax_values.append(0.0)
                missing_reactions.append(reaction_id)
        
        integrated_df['EROG_binary'] = erog_binary_values
        integrated_df['RGR'] = rgr_values
        integrated_df['growth_ko'] = growth_ko_values
        integrated_df['chokepoint_c1'] = chokepoint_values
        integrated_df['flux_min_c1'] = flux_min_values
        integrated_df['flux_max_c1'] = flux_max_values
        integrated_df['is_essential_c1'] = is_essential_values
        integrated_df['reaction_class_c1'] = reaction_class_values
        integrated_df['NR_c1'] = nr_c1_values
        integrated_df['DR_c1'] = dr_c1_values
        integrated_df['RR_c1'] = rr_c1_values
        integrated_df['diff_from_lmax'] = diff_from_lmax_values
        
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
        
        print(f"\n--- Estatísticas da Integração ---")
        print(f"Total de reações no dataset: {len(integrated_df)}")
        print(f"Reações encontradas no modelo: {len(found_reactions)}")
        print(f"Reações não encontradas: {len(missing_reactions)}")
        print(f"Reações EROG (binário): {integrated_df['EROG_binary'].sum()}")
        print(f"Reações chokepoint c=1: {integrated_df['chokepoint_c1'].sum()}")
        print(f"Reações essenciais (c=1): {integrated_df['is_essential_c1'].sum()}")
        
        return integrated_df

    def generate_detailed_report(self, integrated_df: pd.DataFrame, output_path: str = "Pf_CONTRABASS_c1_detailed_report.txt"):
        """Gera relatório detalhado da análise CONTRABASS c=1."""
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("RELATÓRIO DETALHADO - CONTRABASS c=1 (EROG) com Extensão RGR")
        report_lines.append("Implementação conforme: Oarga et al. Bioinformatics, 39(2), 2023")
        report_lines.append("Organismo: Plasmodium falciparum (Modelo: iAM-Pf480)")
        report_lines.append("="*80)
        
        report_lines.append(f"\n1. INFORMAÇÕES DO MODELO")
        report_lines.append(f" Modelo: {self.model_path}")
        report_lines.append(f" Reações no modelo: {len(self.model.reactions)}")
        report_lines.append(f" Metabólitos no modelo: {len(self.model.metabolites)}")
        report_lines.append(f" Crescimento máximo (lmax): {self.lmax:.6f}")
        report_lines.append(f" Reação de biomassa: {self.biomass_reaction_id}")
        report_lines.append(f" Threshold efetivo (ε_eff): {self.effective_threshold:.8f}")
        
        report_lines.append(f"\n2. RESULTADOS CONTRABASS c=1 (CONFORME ARTIGO)")
        report_lines.append(f" Reações EROG identificadas (ER₁): {self.validation_stats['erog_count']}")
        report_lines.append(f" Reações essenciais (crescimento = 0): {self.validation_stats['essential_count']}")
        report_lines.append(f" Chokepoints em c=1 (CP₁): {len(self.chokepoints_c1)}")
        report_lines.append(f" Non-reversible reactions (NR₁): {len(self.nr_c1)}")
        report_lines.append(f" Dead reactions (DR₁): {len(self.dr_c1)}")
        report_lines.append(f" Reversible reactions (RR₁): {len(self.rr_c1)}")
        
        report_lines.append(f"\n3. COMPARAÇÃO COM ARTIGO ORIGINAL")
        report_lines.append(f" EROG: {self.validation_stats['erog_count']} (artigo: 317, Δ={self.validation_stats['erog_count']-317:+d})")
        report_lines.append(f" Essenciais: {self.validation_stats['essential_count']} (artigo: 192, Δ={self.validation_stats['essential_count']-192:+d})")
        
        report_lines.append(f"\n4. TOP 20 REAÇÕES EROG POR IMPACTO (RGR) - Dataset")
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
                
                flag_str = f" [{', '.join(flags)}]" if flags else ""
                report_lines.append(f" {row['label']}: RGR={row['RGR']:.4f}, "
                                  f"growth_ko={row['growth_ko']:.4f}{flag_str}")
        
        report_lines.append(f"\n5. TOP 10 ALVOS PRIORITÁRIOS (EROG + CHOKEPOINT) - Dataset")
        if all(col in integrated_df.columns for col in ['EROG_binary', 'chokepoint_c1', 'RGR']):
            priority_targets = integrated_df[(integrated_df['EROG_binary'] == 1) &
                                            (integrated_df['chokepoint_c1'] == 1)]
            priority_targets = priority_targets.nlargest(10, 'RGR')
            
            for idx, row in priority_targets.iterrows():
                exp_ess = " (exp_essential)" if 'essentiality' in row and row['essentiality'] == 1 else ""
                report_lines.append(f" {row['label']}: RGR={row['RGR']:.4f}{exp_ess}")
        
        report_lines.append("\n" + "="*80)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\nRelatório detalhado salvo em: {output_path}")

    def save_additional_results(self, integrated_df: pd.DataFrame):
        """Salva resultados adicionais em arquivos Excel separados - CORRIGIDO para incluir TODAS as reações do modelo"""
        
        prefix = "Pf_iAM480_"
        
        # 1. Salvar TODAS as reações EROG do modelo
        all_erog_data = []
        for rid, data in self.erog_results.items():
            if data['is_erog']:
                dataset_info = {}
                if rid in integrated_df['label'].values:
                    row = integrated_df[integrated_df['label'] == rid].iloc[0]
                    dataset_info = {
                        'essentiality_exp': row.get('essentiality', 'N/A'),
                        'fba_pred': row.get('fba_pred', 'N/A'),
                        'in_dataset': True
                    }
                else:
                    dataset_info = {
                        'essentiality_exp': 'N/A',
                        'fba_pred': 'N/A',
                        'in_dataset': False
                    }
                
                all_erog_data.append({
                    'reaction_id': rid,
                    'rgr': data['rgr'],
                    'growth_ko': data['growth_ko'],
                    'diff_from_lmax': data.get('diff_from_lmax', 0.0),
                    'is_essential': data['is_essential'],
                    'reaction_class_c1': data.get('reaction_class_c1', 'UNKNOWN'),
                    'is_chokepoint': rid in self.chokepoints_c1,
                    'flux_min_c1': data.get('flux_min_c1', 0.0),
                    'flux_max_c1': data.get('flux_max_c1', 0.0),
                    **dataset_info
                })
        
        erog_df = pd.DataFrame(all_erog_data)
        erog_df = erog_df.sort_values('rgr', ascending=False)
        erog_df.to_excel(f"{prefix}CONTRABASS_c1_all_EROG_reactions.xlsx", index=False)
        print(f" ✅ TODAS as reações EROG ({len(all_erog_data)}) salvas em: {prefix}CONTRABASS_c1_all_EROG_reactions.xlsx")
        
        # 2. Salvar reações prioritárias (EROG + chokepoint)
        priority_data = []
        for rid, data in self.erog_results.items():
            if data['is_erog'] and rid in self.chokepoints_c1:
                dataset_info = {}
                if rid in integrated_df['label'].values:
                    row = integrated_df[integrated_df['label'] == rid].iloc[0]
                    dataset_info = {
                        'essentiality_exp': row.get('essentiality', 'N/A'),
                        'in_dataset': True
                    }
                else:
                    dataset_info = {
                        'essentiality_exp': 'N/A',
                        'in_dataset': False
                    }
                
                priority_data.append({
                    'reaction_id': rid,
                    'rgr': data['rgr'],
                    'growth_ko': data['growth_ko'],
                    'is_essential': data['is_essential'],
                    'reaction_class_c1': data.get('reaction_class_c1', 'UNKNOWN'),
                    'flux_min_c1': data.get('flux_min_c1', 0.0),
                    'flux_max_c1': data.get('flux_max_c1', 0.0),
                    **dataset_info
                })
        
        priority_df = pd.DataFrame(priority_data)
        priority_df = priority_df.sort_values('rgr', ascending=False)
        priority_df.to_excel(f"{prefix}CONTRABASS_c1_priority_targets.xlsx", index=False)
        print(f" ✅ Alvos prioritários ({len(priority_data)}) salvos em: {prefix}CONTRABASS_c1_priority_targets.xlsx")
        
        # 3. Salvar TODAS as reações essenciais
        essential_data = []
        for rid, data in self.erog_results.items():
            if data['is_essential']:
                dataset_info = {}
                if rid in integrated_df['label'].values:
                    row = integrated_df[integrated_df['label'] == rid].iloc[0]
                    dataset_info = {
                        'essentiality_exp': row.get('essentiality', 'N/A'),
                        'in_dataset': True
                    }
                else:
                    dataset_info = {
                        'essentiality_exp': 'N/A',
                        'in_dataset': False
                    }
                
                essential_data.append({
                    'reaction_id': rid,
                    'rgr': data['rgr'],
                    'growth_ko': data['growth_ko'],
                    'reaction_class_c1': data.get('reaction_class_c1', 'UNKNOWN'),
                    'is_chokepoint': rid in self.chokepoints_c1,
                    **dataset_info
                })
        
        essential_df = pd.DataFrame(essential_data)
        essential_df.to_excel(f"{prefix}CONTRABASS_c1_essential_reactions.xlsx", index=False)
        print(f" ✅ Reações essenciais ({len(essential_data)}) salvas em: {prefix}CONTRABASS_c1_essential_reactions.xlsx")
        
        # 4-6. Salvar classificações
        for name, data_set, filename in [
            ('NR₁', self.nr_c1, f"{prefix}CONTRABASS_c1_NR_reactions.xlsx"),
            ('DR₁', self.dr_c1, f"{prefix}CONTRABASS_c1_DR_reactions.xlsx"),
            ('RR₁', self.rr_c1, f"{prefix}CONTRABASS_c1_RR_reactions.xlsx")
        ]:
            df = pd.DataFrame({'reaction_id': list(data_set)})
            df.to_excel(filename, index=False)
            print(f" ✅ Reações {name} ({len(data_set)}) salvas em: {filename}")
        
        # 7. Salvar estatísticas resumidas
        stats_data = {
            **self.validation_stats,
            'total_reactions_in_model': len(self.model.reactions),
            'total_metabolites_in_model': len(self.model.metabolites),
            'biomass_reaction_id': self.biomass_reaction_id,
            'erog_in_dataset': sum(1 for rid in self.erog_results if self.erog_results[rid]['is_erog'] and rid in integrated_df['label'].values),
            'erog_not_in_dataset': sum(1 for rid in self.erog_results if self.erog_results[rid]['is_erog'] and rid not in integrated_df['label'].values),
            'target_erog_article': 317,
            'erog_diff_from_article': self.validation_stats['erog_count'] - 317,
        }
        stats_df = pd.DataFrame([stats_data])
        stats_df.to_excel(f"{prefix}CONTRABASS_c1_summary_statistics.xlsx", index=False)
        print(f" ✅ Estatísticas resumidas salvas em: {prefix}CONTRABASS_c1_summary_statistics.xlsx")
        
        # 8. Salvar lista de chokepoints
        choke_list = pd.DataFrame(list(self.chokepoints_c1), columns=['reaction_id'])
        choke_list.to_excel(f"{prefix}CONTRABASS_c1_chokepoints_list.xlsx", index=False)
        print(f" ✅ Lista de chokepoints ({len(self.chokepoints_c1)}) salva em: {prefix}CONTRABASS_c1_chokepoints_list.xlsx")
        
        # 9. Salvar classificação completa
        classification_details = []
        for rid, data in self.erog_results.items():
            classification_details.append({
                'reaction_id': rid,
                'is_erog': data['is_erog'],
                'rgr': data['rgr'],
                'growth_ko': data['growth_ko'],
                'diff_from_lmax': data.get('diff_from_lmax', 0.0),
                'is_essential': data['is_essential'],
                'flux_min_c1': data.get('flux_min_c1', 0.0),
                'flux_max_c1': data.get('flux_max_c1', 0.0),
                'reaction_class_c1': data.get('reaction_class_c1', 'UNKNOWN'),
                'is_chokepoint_c1': rid in self.chokepoints_c1,
                'in_nr_c1': rid in self.nr_c1,
                'in_dr_c1': rid in self.dr_c1,
                'in_rr_c1': rid in self.rr_c1,
                'in_dataset': rid in integrated_df['label'].values
            })
        
        class_df = pd.DataFrame(classification_details)
        class_df.to_excel(f"{prefix}CONTRABASS_c1_complete_classification.xlsx", index=False)
        print(f" ✅ Classificação completa de {len(classification_details)} reações salva em: {prefix}CONTRABASS_c1_complete_classification.xlsx")
        
        # Resumo de cobertura
        erog_count = len(all_erog_data)
        if erog_count > 0:
            print(f"\n📊 RESUMO DE COBERTURA:")
            print(f"• Reações EROG no modelo: {erog_count}")
            print(f"• Reações EROG no dataset: {stats_data['erog_in_dataset']}")
            print(f"• Reações EROG fora do dataset: {stats_data['erog_not_in_dataset']}")
            print(f"• Cobertura do dataset: {stats_data['erog_in_dataset']/erog_count*100:.1f}%")

def main():
    """Função principal para executar a análise CONTRABASS c=1."""
    
    MODEL_PATH = "iAM_Pf480.json" 
    EXISTING_CSV_PATH = "iML1515_mfg_nodes_ess-label_fba_pred.csv"
    OUTPUT_PATH = "Pf_iAM480_CONTRABASS_c1_results.csv"
    REPORT_PATH = "Pf_CONTRABASS_c1_detailed_report.txt"
    
    try:
        print("="*80)
        print("CONTRABASS c=1 - Análise de Vulnerabilidades Metabólicas")
        print("Implementação conforme: Oarga et al. Bioinformatics, 39(2), 2023")
        print("Organismo: Plasmodium falciparum (Modelo: iAM-Pf480)")
        print("Foco: EROG (ER₁) - Essential Reactions for Optimal Growth (c=1)")
        print("Threshold: Relativo (0.01% do lmax) + Absoluto (1e-6)")
        print("="*80)
        
        analyzer = CONTRABASS_EROG(MODEL_PATH, EXISTING_CSV_PATH)
        
        print("\n[1/4] Carregando dados e executando CONTRABASS c=1...")
        analyzer.load_and_validate_data()
        
        print("\n[2/4] Integrando resultados CONTRABASS...")
        integrated_df = analyzer.integrate_erog_data()
        
        integrated_df.to_csv(OUTPUT_PATH, index=False)
        print(f"\nResultados principais salvos em: {OUTPUT_PATH}")
        
        print("\n[3/4] Gerando relatório analítico CONTRABASS...")
        analyzer.generate_detailed_report(integrated_df, REPORT_PATH)
        
        print("\n[4/4] Salvando resultados adicionais...")
        analyzer.save_additional_results(integrated_df)
        
        print("\n" + "="*80)
        print("ANÁLISE CONTRABASS c=1 CONCLUÍDA COM SUCESSO!")
        print("="*80)
        
        print("\n📊 ARQUIVOS GERADOS:")
        print("1. Pf_iAM480_CONTRABASS_c1_results.csv")
        print("2. Pf_iAM480_CONTRABASS_c1_all_EROG_reactions.xlsx")
        print("3. Pf_iAM480_CONTRABASS_c1_priority_targets.xlsx")
        print("4. Pf_iAM480_CONTRABASS_c1_essential_reactions.xlsx")
        print("5. Pf_iAM480_CONTRABASS_c1_NR_reactions.xlsx")
        print("6. Pf_iAM480_CONTRABASS_c1_DR_reactions.xlsx")
        print("7. Pf_iAM480_CONTRABASS_c1_RR_reactions.xlsx")
        print("8. Pf_iAM480_CONTRABASS_c1_summary_statistics.xlsx")
        print("9. Pf_iAM480_CONTRABASS_c1_chokepoints_list.xlsx")
        print("10. Pf_iAM480_CONTRABASS_c1_complete_classification.xlsx")
        print("11. Pf_CONTRABASS_c1_detailed_report.txt")
        
    except Exception as e:
        print(f"\n❌ ERRO na análise CONTRABASS: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()