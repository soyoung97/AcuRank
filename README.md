# AcuRank: Uncertainty-Aware Adaptive Computation for Listwise Reranking

Official code and baselines for our [AcuRank](http://arxiv.org/abs/2505.18512) paper. 

This repository includes our implementation of AcuRank as well as baseline methods. Listwise reranker prompting, generation, and parsing are based on [RankLLM](https://github.com/castorini/rank_llm). The TourRank baseline is adapted from [TourRank](https://github.com/chenyiqun/TourRank).

![AcuRank](main_figure.png)

---

## 1. Quick Start

```bash
# Run AcuRank on TREC-COVID (BM25 top-100)
CUDA_VISIBLE_DEVICES=0 \
python run.py \
  --method trueskill \
  --R 10 \
  --uncertain_U 10 \
  --tol 1e-2 \
  --use_firststage_orderings \
  --break_mode reduce_uncertain \
  --dataset trec-covid
```

All variant commands below assume the same flags unless specified otherwise.

---

## 2. Dataset Preparation

We provide **BM25 top-100** retrievals for three datasets:

| ID           | Description  |
| ------------ | ------------ |
| `trec-covid` | TREC-COVID   |
| `dl23`       | TREC-DL 2023 |
| `dl-hard`    | DL-Hard      |

Specify the dataset with `--dataset <name>`.

> **Custom data \& Other datasets**
> Convert and make other retrieval runs to the required JSONL format, by running & modifying the `make_jsonl_data.py` script.
> Other datasets can be downloaded via the pre-dumped [huggingface datasets](https://huggingface.co/datasets/Soyoung97/beir-eval-bm25-top100) repository.

---

## 3. Directory Layout

```text
AcuRank/
├── adaptive_utils.py              # Dynamic Programming solver for threshold t s.t. P(x_i > t)=R
├── beir_eval.py                   # NDCG@10 evaluator
├── listwise_reranking_modules.py  # Prompting, generation, parsing utilities
├── run.py                         # Entry point for AcuRank / SW / TS variants
├── tourrank.py                    # Baseline: TourRank implementation
├── make_jsonl_data.py             # Convert first-stage runs to JSONL
└── data/                          # Stores first-stage retrieval files
    └── bm25/
        └── ...
```

---

## 4. Environment

```text
python          == 3.10.13
torch           == 2.6.0
transformers    == 4.50.3   # (4.40.0 for RankVicuna)
beir            == 2.1.0
trueskill       == 0.4.5
sentencepiece   == 0.2.0
jsonlines, pandas, scipy, ftfy, …
```

Add any missing packages if prompted during setup.

---

## 5. AcuRank Variants

| Variant        | Additional / Modified Flags  |
| -------------- | ---------------------------- |
| **AcuRank-9**  | `--hard_constraint 4`        |
| **AcuRank-H**  | `--tol 1e-4`                 |
| **AcuRank-HH** | `--tol 1e-4 --uncertain_U 5` |

---

## 6. Ablations (Table 3)

| Design Choice          | Flag Change                         |
| ---------------------- | ----------------------------------- |
| *No first-stage init*  | remove `--use_firststage_orderings` |
| *Random chunking*      | add `--chunking_mode random`        |
| *Top-k stability stop* | change `--break_mode top10_nochange`       |

---

## 7. Baselines

### 7.1 Sliding Windows (SW-X)

```bash
# SW-1 (one pass)
CUDA_VISIBLE_DEVICES=0 \
python run.py --method sliding_windows --num_pass 1 --dataset trec-covid
```

Use `--num_pass` to set the number of passes (e.g. 2 or 3 for SW-2 / SW-3).

### 7.2 TourRank (TourRank-X)

```bash
# TourRank-1
CUDA_VISIBLE_DEVICES=0 \
python tourrank.py --dataset trec-covid --rep 1
```

Increase `--rep` (e.g. 5) for TourRank-5.

### 7.3 TrueSkill-Static (TS-X)

```bash
# TS-10 (budget 5-2-2-1)
CUDA_VISIBLE_DEVICES=0 \
python run.py \
  --method fixed_budget \
  --use_firststage_orderings \
  --budget_per_stage 5 2 2 1 \
  --dataset trec-covid
```

Change the sequence for other budgets (e.g. `5 4 4 4 4 4` for TS-25).

---

## 8. Using Different Rerankers

| Reranker          | How to Enable                                                                            |
| ----------------- | ---------------------------------------------------------------------------------------- |
| **RankVicuna-7B** | `--model_path castorini/rank_vicuna_7b_v1`                                               |
| **GPT-4.1-mini**  | add your API key in `listwise_reranking_modules.py` then set `--model_path gpt-4.1-mini` |
| **Llama-3.3-70B-Instruct** | `--model_path meta-llama/Llama-3.3-70B-Instruct`  |

---


## Citations

If you find our work useful, please consider citing our paper:

```
@article{yoon2025acurank,
  title={AcuRank: Uncertainty-Aware Adaptive Computation for Listwise Reranking},
  author={Yoon, Soyoung and Kim, Gyuwan and Cho, Gyu-Hwung and Hwang, Seung-won},
  journal={arXiv preprint arXiv:2505.18512},
  year={2025}
}
```


---

## Contact

If you have any questions or run into issues, feel free to open a GitHub Issue.  
Please email `soyoung.yoon@snu.ac.kr` if you have any further questions about the implementation.  
You can also email `gyuwankim@ucsb.edu` with questions about the methodology or the paper.


---

## License

To be updated.

