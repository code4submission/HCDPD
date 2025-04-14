HCDPD (Heterogeneous Causal Disease Pattern Detection) is a Python coding based package for mapping the complex causal pathways 
from early-stage diseases to latent disease patterns and their manifestation in organs as observed in later-stage medical images.

# Command Script

## gen_simu_data.py
- generate simulation dataset
- command line: python gen_simu_data.py --id 1   (1 is the simulation id for random seed specification)
- dependence files
  functions.py (all functions used in main commandlines)
  ./dat/real_data (the simulated datasets are generated based on the real dataset)
- output (simulated data files saved in .npy format in folder ./dat/simu_data/1, 1 is the simulation id for random seed specification)
  
## run_simu_data.py
- run simulation studies
- command line: python run_simu_data.py --id    (1 is the simulation id for random seed specification)
- dependence files
  functions.py (all functions used in main commandlines)
  ./dat/real_data (the simulated datasets are generated based on the real dataset)
- output (simulated data files saved in .npy format in folder ./dat/simu_data/1, 1 is the simulation id for random seed specification)
