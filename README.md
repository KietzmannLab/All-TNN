# End-to-end topographic networks (All-TNNs) as models of cortical map formation and human visual behaviour
**Authors: Zejin Lu, Adrien Doerig, Victoria Bosch, Bas Krahmer, Daniel Kaiser, Radoslaw M Cichy, & Tim C Kietzmann**

🔗 You can find our preprint [here](https://arxiv.org/pdf/2308.09431) 🔗

### Abstract
*A prominent feature of the primate visual system is its topographical organisation. For understanding its origins, its computational role, and its behavioural implications, computational models are of central importance. Yet, vision is commonly modelled using convolutional neural networks which are hard-wired to learn identical features across space and thus lack topography. Here, we overcome this limitation by introducing All-Topographic Neural Networks (All-TNNs). All-TNNs develop several features reminiscent of primate topography, including smooth orientation and category selectivity maps, and enhanced processing of regions with task-relevant information. In addition, All-TNNs operate on a low energy budget, suggesting a metabolic benefit of smooth topographic organisation. To test our model against behaviour, we collected a novel dataset of human spatial biases in object recognition and found that All-TNNs significantly outperform control models. All-TNNs thereby offer a promising candidate for modelling of primate visual topography and its role in downstream behaviour.*

## Data and model checkpoints 
Model weight checkpoints can be found in the [OSF directory](https://osf.io/6m3g4/?) and can be loaded using `test_model.py`. 

Human and model behavioural data, as well as analysis results (such as model activations on the test set) can be found in the [OSF directory](https://osf.io/6m3g4/?view_only=2950b15542c84d7ca53a7312238a2980) as well. 
This repository can also be used to generate these data and results (see `analysis.py`). 

The Ecoset dataset can be found on [HuggingFace](https://huggingface.co/datasets/kietzmannlab/ecoset)🤗.


## How to Use This Repository

This repository, available at [KietzmannLab/All-TNN](https://github.com/KietzmannLab/All-TNN), contains the implementation of All-Topographic Neural Networks (All-TNNs), along with the analysis code for exploring cortical map formation (e.g. orientation selectivity, category selectivity), energy efficiency, spatial biases in visual object recognition and comparisons with human visual behaviour. The following sections will guide you through the setup, usage, and detailed exploration of the All-TNN models.

   Download the analysis data from the storage into `save_dir` to be able to visualize all figures, or generate the results yourself using `analysis.py` (see below). 
   **To reproduce the main figures in the paper, use `plot_main_analysis.ipynb`** 

### Prerequisites

- Python >= 3.8
- TensorFlow 2.12
- Additional Python libraries as specified in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/KietzmannLab/All-TNN.git
   ```
2. Navigate to the repository directory and pip install All-TNN as a package:
   ```bash
   cd All-TNN
   pip install -e .
   ```
3. Create and activate your environment
   ```bash
   python3 venv -m all_tnn_env
   source all_tnn_env/bin/activate
   ```
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---
### Analyzing Models

To generate results for multiple models, adjust the `config.py` in the folder `all_tnn/analysis`. Here you can select models and model seeds to analyze, as well as which analyses to perform. 

```shell
python scripts/analysis.py 
```

**Note:** computing several of these analyses can be very time-consuming, in particular retrieving model activations on test datasets. The results of all analyses can be downloaded pre-computed at the `save_dir` in the [OSF repository](https://osf.io/6m3g4/?view_only=2950b15542c84d7ca53a7312238a2980)
Memory requirements: <300GB. For calculating the energy consumption of a model on the test set, 500GB is required, because all activations of all layers on the test set need to be kept in memory. 

### Loading and Testing Models

To test a pretrained model on new stimuli, run `scripts/model_test.py`. 
Specify the model name and epoch and preprocess stimuli from your dataset of choice. 

---


## Contributions and Feedback

We welcome contributions and feedback from the research community. If you would like to contribute or have any suggestions or questions, please feel free to open an issue or submit a pull request.

## Citation

If you use All-TNNs or any of the provided datasets in your research, please cite our paper:
``` 
Lu, Z., Doerig, A., Bosch, V., Krahmer, B., Kaiser, D., Cichy, R. M., & Kietzmann, T. C. (2023). End-to-end topographic networks as models of cortical map formation and human visual behaviour: moving beyond convolutions. arXiv preprint arXiv:2308.09431.
```

```bibtex
@article{lu2023end,
  title={End-to-end topographic networks as models of cortical map formation and human visual behaviour: moving beyond convolutions},
  author={Lu, Z. and Doerig, A. and Bosch, V. and Krahmer, B. and Kaiser, D. and Cichy, R. M. and Kietzmann, T. C.},
  journal={arXiv preprint arXiv:2308.09431},
  year={2023}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
