# Experiment Results — ThinX
This branch of the **ThinX** repository contains the results of experiments related to computing model explanations on various datasets from OpenML (CC18 and CTR23 benchmark suits) using different distribution compression methods, as well as scripts used for running computations on a SLURM cluster (EDEN).

```text
package_metadata/
├── openml/                                  
│   ├── {dataset_name}/                      # Example: abalone, adult, bank-marketing...
│   │   ├── custom_kt_methods/               
│   │   │   └── kernel_thinning_{data_modification_method}_{explainer_name}_{strategy}_{model_name}_{seed}_{repeats_number}.csv
│   │   │
│   │   ├── ground_truth/                   
│   │   │   └── {explainer_name}_{strategy}_{repeats_number}_{model_name}.npz
│   │   │
│   │   ├── kernels_iid_comparison/         
│   │   │   └── {explainer_name}_{strategy}_{model_name}_kernels_iid_{seed}_{repeats_number}.csv
│   │   │
│   │   └── other_baselines/                 
│   │       └── {compression_method}_none_{explainer_name}_{strategy}_{model_name}_{seed}_{repeats_number}.csv
│   │
│   ├── cc18_information.csv                 # Metadata for classification datasets and models
│   └── ctr23_information.csv                 # Metadata for regression datasets and models
│
└── scripts/                                 # --- EXECUTABLE SCRIPTS ---
    ├── scripts_custom_kt_methods/           # Experiments with custom KT modifications
    │   ├── custom_kt_methods.py             # Core experiment logic functions
    │   └── runner_custom_kt_methods.py      # SLURM job submission generator
    │
    ├── scripts_ground_truth/                # Scripts for Ground Truth experiment generation
    ├── scripts_kernels/                     # Scripts for kernel comparison experiments
    └── scripts_other_baselines/             # Scripts for experiments comparing baselines
```

## Notes on Experimental Setup and Reproducibility

- Each experiment was designed to run **10 repetitions**. Due to time limits imposed by the SLURM scheduler, repetitions were sometimes executed in **separate batches** (e.g., 1, 2, or 4 repetitions per job). This is reflected in the stored result files through the **random seed values** and the **number of repetitions** encoded in file names.
- Memory requirements were **dataset-dependent**. For most datasets, memory limits were reduced to optimize cluster resource usage, while for certain datasets higher memory allocations were required. All memory configurations are explicitly specified in the corresponding SLURM job scripts.
