HCDPD (Heterogeneous Causal Disease Pattern Detection) is a Python coding based package for mapping the complex causal pathways 
from early-stage diseases to latent disease patterns and their manifestation in organs as observed in later-stage medical images.

# Command Script

## gen_simu_data.py
- generate simulation dataset
- command line: python gen_simu_data.py --id 1   (1 is the simulation id for random seed specification)
- dependence files:
- functions.py (all functions used in main commandlines)
- ./dat/real_data (the simulated datasets are generated based on the real dataset)
- output:
- simulated data files saved in .npy format in folder ./dat/simu_data/1 (1 is the simulation id for random seed specification)
  
## run_simu_data.py
- run simulation studies
- command line: python run_simu_data.py --id 1  (1 is the simulation id for random seed specification)
- dependence files:
- functions.py (all functions used in main commandlines)
- ./dat/simu_data/1 (1 is the simulation id for random seed specification)
- output:
- all the MCMC samples for parameters and disease regions
- files saved in .npy format in folder ./dat/simu_result/simu_1 (1 is the simulation id for random seed specification)

## report_simu_data.py
- report simulation studies results 
- command line: python report_simu_data.py --id 1  (1 is the simulation id for random seed specification)
- dependence files:
- functions.py (all functions used in main commandlines)
- ./dat/simu_result/simu_1 (1 is the simulation id for random seed specification)
- output:
- Figures for results based on simulation dataset 1 (1 is the simulation id for random seed specification)

- ## summary_simu_data.py
- summarize performance across all simulation studies results 
- command line: python summary_simu_data.py --p 0.5  (p is the threshold value for determining the disease region)
- dependence files:
- functions.py (all functions used in main commandlines)
- all simulation results {./dat/simu_result/simu_ids} (ids: simulation ids)
- output:
- Figures for results based on all simulation results

- ## run_real_data.py
- run real data analysis
- command line: python run_real_data.py 
- dependence files:
- functions.py (all functions used in main commandlines)
- ./dat/real_data
- output:
- all the MCMC samples for parameters and disease regions
- files saved in .npy format in folder ./dat/real_result

- ## report_real_data.py
- report real data analyses results 
- command line: python report_real_data.py
- dependence files:
- functions.py (all functions used in main commandlines)
- ./dat/real_result
- output:
- Figures for real data analyses results 
