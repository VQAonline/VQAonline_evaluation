# Evaluation for VQAonline
Evaluation code for VQAonline, which includes the setup of LLama, METEOR, BERTscore, ROUGEL, CLIPScore, and RefCLIPScore. 

## LLama evaluation
1. Set up: Please follow the "Quick Start" for the [LLama2 setup](https://github.com/meta-llama/llama).
2. Example script: You can run the example script for LLama evaluation with the following code:

`torchrun --nproc_per_node 2 llama_eval.py`
## METEOR, BERTscore, and ROUGEL evaluation
1. For BERTscore, we used the implementation from [GEM-metric](https://github.com/GEM-benchmark/GEM-metrics). You can set up it with:
  
  ```
  git clone https://github.com/GEM-benchmark/GEM-metrics
  cd GEM-metrics
  pip install -r requirements.txt -r requirements-heavy.txt
  ```

2. For METEOR and ROUGEL evaluation, we used the implementation from [pycocoevalcap](https://github.com/salaniz/pycocoevalcap). Please install pycocoevalcap and the pycocotools dependency (https://github.com/cocodataset/cocoapi) by running:

```pip install pycocoevalcap```

3. Example script for METEOR, BERTscore, and ROUGEL evaluation:
   
```python other_metrics.py```

## CLIPScore and RefCLIPScore evaluation
Please follow [clipscore](https://github.com/jmhessel/clipscore.git) to run CLIPscore and RefCLIPScore.
## Citation
```
@inproceedings{chen2024fully,
  title={Fully authentic visual question answering dataset from online communities},
  author={Chen, Chongyan and Liu, Mengchen and Codella, Noel and Li, Yunsheng and Yuan, Lu and Gurari, Danna},
  booktitle={European Conference on Computer Vision},
  pages={252--269},
  year={2024},
  organization={Springer}
}

```
