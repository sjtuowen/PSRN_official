
## 📥 Installation

### Step 1: Create an Environment and Install PyTorch

```bash
conda env create -n PSRNpip python=3.12
```

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install uv
uv pip install numpy numba deap==1.3.3 click PyYAML symengine==0.11.0
uv pip install tqdm scipy
uv pip install scikit-learn
```

Package                  Version
------------------------ ------------
click                    8.3.1
deap                     1.3.3
filelock                 3.19.1
fsspec                   2025.9.0
Jinja2                   3.1.6
llvmlite                 0.45.1
MarkupSafe               2.1.5
mpmath                   1.3.0
networkx                 3.5
numba                    0.62.1
numpy                    2.3.3
nvidia-cublas-cu12       12.8.4.1
nvidia-cuda-cupti-cu12   12.8.90
nvidia-cuda-nvrtc-cu12   12.8.93
nvidia-cuda-runtime-cu12 12.8.90
nvidia-cudnn-cu12        9.10.2.21
nvidia-cufft-cu12        11.3.3.83
nvidia-cufile-cu12       1.13.1.3
nvidia-curand-cu12       10.3.9.90
nvidia-cusolver-cu12     11.7.3.90
nvidia-cusparse-cu12     12.5.8.93
nvidia-cusparselt-cu12   0.7.1
nvidia-nccl-cu12         2.27.5
nvidia-nvjitlink-cu12    12.8.93
nvidia-nvshmem-cu12      3.3.20
nvidia-nvtx-cu12         12.8.90
pillow                   11.3.0
pip                      25.3
PyYAML                   6.0.3
scipy                    1.16.3
setuptools               80.9.0
symengine                0.11.0
sympy                    1.14.0
torch                    2.9.1+cu128
torchvision              0.24.1+cu128
tqdm                     4.67.1
triton                   3.5.1
typing_extensions        4.15.0
uv                       0.9.10
wheel                    0.45.1


----------------🥝🥝🥝🥝🥝🥝🥝🥝🥝🥝🥝🥝🥝🥝🥝🥝


### Step 2: Install Other Dependencies Using Pip

```bash
conda activate PSRN
pip install -r requirements.txt
```

### ⚠️ Important Notes

- If using a version of PyTorch below 2.0, an error may occur during the `torch.topk` operation.
- The experiments were performed on servers with Nvidia A100 (80GB) and Intel(R) Xeon(R) Platinum 8380 CPUs @ 2.30GHz.
- We recommend using a high-memory GPU as smaller cards may encounter CUDA memory errors under our experimental settings. If you experience memory issues, consider reducing the number of input slots or opting for `semi_koza` operator sets (e.g., replacing `"Sub"` and `"Div"` with `"SemiSub"` and `"SemiDiv"`) or `basic` operator sets (e.g., replacing `"Sub"` and `"Div"` with `"Neg"` and `"Inv"`).

## 🚀 Quickstart with Custom Data

To execute the script with custom data, use the following arguments:

- `-g`: Specifies the GPU to use. Enter the GPU index.
- `-i`: Sets the number of input slots for PSRN.
- `-c`: Indicates whether to include constants in the computation (True / False).
- `-l`: Defines the operator library to be used. Specify the name of the library or an operator list.
- `--csvpath`: Specifies the path to the CSV file to be used. By default, if not specified, it uses `./custom_data.csv`. Each column represents an independent variable.

For more detailed parameter settings, please refer to the `run_custom_data.py` script.

### 📝 Examples

To run the script with custom data with an expression probe (the algorithm will stop when it finds the expression or its symbolic equivalents):

```bash
python run_custom_data.py -g 0 -i 5 -c False --probe "(exp(x)-exp(-x))/2"
```

Without an expression probe:

```bash
python run_custom_data.py -g 0 -i 5 -c False
```

For limited VRAM (or when the ground truth expression is expected to be simple):

```bash
python run_custom_data.py -g 0 -i 3 -c False --probe "(exp(x)-exp(-x))/2"
```

To customize the operator library:

```bash
python run_custom_data.py -g 0 -i 5 -c False --probe "(exp(x)-exp(-x))/2" -l "['Add','Mul','Identity','Tanh','Abs']"
```
_Note: If using GP as the token generator, you may also need to change the operator that GP uses in `token_generator_config.yaml`_

For custom data paths:

```bash
python run_custom_data.py -g 0 -i 5 -c False --probe "(exp(x)-exp(-x))/2" --csvpath ./another_custom_data.csv
```

### 📋 Note on DR Mask Files

The `.npy` files under `./dr_mask` are pre-generated. When you try to use a new network architecture (e.g., a new combination of operators, number of variables, and number of layers), you may need to run the gen_dr_mask.py script first. Typically, this process takes less than a minute:

```bash
python utils/gen_dr_mask.py --n_symbol_layers=3 --n_inputs=5 --ops="['Add','Mul','SemiSub','SemiDiv','Identity','Sin','Cos','Exp','Log','Tanh','Cosh','Abs','Sign']"
```

## ⚙️ Custom Stages

If you want to customize the operators, the number of PSRN inputs, and configure custom search stages, you can edit the `.yaml` files under the `model/stages_config` directory, or pass a dictionary directly to the `stage_config=...` field when instantiating the `PSRN_Regressor`.

Example:

```yaml
default:
  operators: ['Add', 'Mul', 'SemiSub', 'SemiDiv', 'Identity', 'Sin', 'Cos', 'Exp', 'Log']
  time_limit: 900  # 60 * 15
  n_psrn_inputs: 7
  n_sample_variables: 5

stages:
  - {time_limit: 5, n_psrn_inputs: 5}
  - {}
  - {}
```

Explanation: First, set the default stage parameters in `default`. Then, design the specific search workflow in `stages`. An empty pair of curly braces `{}` indicates a stage that uses the default parameters.

## 📊 Symbolic Regression Benchmark

To reproduce our experiments, execute the following command:

```bash
python run_benchmark_all.py --n_runs 100 -g 0 -b benchmark.csv
```

For the Feynman expressions:

```bash
python run_benchmark_all.py --n_runs 100 -g 0 -b benchmark_Feynman.csv
```

The Pareto optimal expressions and corresponding statistics for each puzzle are available in the `log/benchmark` directory. Additionally, the expected runtime for each puzzle can be found in the supplementary materials.

## 🔄 Chaotic Dynamics

Discover the dynamics of chaotic systems by running:

```bash
python run_chaotic.py --n_runs 50 -g 0     # Using GPU index 0
```

This script will generate Pareto optimal expressions for each derivative, and the outcomes will be stored in the `log/chaotic` directory.

### 📈 Evaluating Symbolic Recovery

Assess the symbolic recovery rate by executing:

```bash
python result_analyze_chaotic.py
```

This analysis will automatically compute and save the statistics to `log/chaotic_symbolic_recovery/psrn_stats.csv`

## 🔬 Realworld Data - EMPS

```bash
python run_realworld_EMPS.py --n_runs 20 -g 0    # Using GPU index 0
```

The results (Pareto optimal expressions) can be found in `log/EMPS`

## 🔬 Realworld Data - Turbulent Friction

```bash
python run_realworld_roughpipe.py --n_runs 20 -g 0     # Using GPU index 0
```

The results (Pareto optimal expressions) can be found in `log/roughpipe`

## 🧪 Ablation Studies

To reproduce our ablation studies, execute the following commands.
The results will be stored in the `log/` directory.

### Token Generator Ablation

```bash
python run_benchmark_all.py --n_runs 100 -g 0 -b benchmark.csv -t MCTS
python run_benchmark_all.py --n_runs 100 -g 0 -b benchmark.csv -t GP
python run_benchmark_all.py --n_runs 100 -g 0 -b benchmark.csv -t Random
```

### Constant Range Sensitivity

```bash
python study_ablation/constants/run_c_experiments.py --n_runs 20 -g 0 -t GP
python study_ablation/constants/run_c_experiments.py --n_runs 20 -g 0 -t MCTS
```

### DR Mask Ablation

You can modify the operator library and adjust the number of input slots in yaml file, and choose whether to use the DR Mask.
While the script is running, monitor the memory footprint using nvidia-smi or nvitop.

```bash
python study_ablation/drmask/run_without_drmask.py --use_drmask True -g 0
python study_ablation/drmask/run_without_drmask.py --use_drmask False -g 0
```

## 🏆 SRbench Evaluation

To evaluate PSE's performance on SRbench, follow these steps:

1. Copy the `SRBenchRegressor` directory into the `algorithm` folder of your SRbench installation.

2. Follow the standard SRbench instructions to run our algorithm.

This setup will allow you to benchmark PSE using the SRbench, providing a standardized evaluation of its performance alongside other symbolic regression algorithms.

## 📚 Citation

If you use this work, please cite:

```bibtex
@article{arxiv:2407.04405,
  author     = {Ruan, Kai and Gao, Ze-Feng and Guo, Yike and Sun, Hao and Wen, Ji-Rong and Liu, Yang},
  title      = {Discovering symbolic expressions with parallelized tree search},
  journal    = {arXiv preprint arXiv:2407.04405},
  year       = {2024},
  url        = {https://arxiv.org/abs/2407.04405}
}
```
