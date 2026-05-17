PowerFlowGAT: Reprodutibilidade e Extensão do FlowGAT
  Este repositório contém os códigos desenvolvidos durante a dissertação de mestrado PowerFlowGAT – Uma Rede de Atenção em Grafos Enriquecida com Preços-Sombra e Métricas de Vulnerabilidade Metabólica para Predição de Essencialidade Gênica.
Aqui você encontrará todos os scripts utilizados para:

1. Reprodução integral do pipeline do FlowGAT original – Geração dos três arquivos de entrada obrigatórios do modelo:

    iML1515_mfg_edges.csv (arestas do grafo de fluxo de massa)
    
    iML1515_mfg_nodes_ess-label_fba_pred.csv (nós com atributos topológicos e rótulos de essencialidade)
    
    mfg_essentialities_indeces.npy (índices dos nós rotulados)

2. Cálculo e validação de preços-sombra: Extração dos multiplicadores duais da FBA, verificação de degenerescência por perturbação direta (brute-force) e consolidação em analise_precos_sombra_iML1515.xlsx.

    Propagação de preços-sombra para reações: Módulo ShadowPrice_Reactions que calcula o preço-sombra médio ponderado de cada reação, gerando ShadowPrice_Reactions.csv.

3. Aplicação do CONTRABASS e cálculo da Redução Relativa do Crescimento (RGR): Identificação das reações essenciais para o crescimento ótimo (EROG) e quantificação contínua do impacto de nocautes sobre a taxa de crescimento.


  Os códigos foram validados contra os modelos e resultados originais: iJO1366 para preços-sombra (Reznik et al., 2013) e iAM‑Pf480 para CONTRABASS (Oarga et al., 2023), antes de serem aplicados ao modelo iML1515 (E. coli K‑12 MG1655).
