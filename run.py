import copy
import logging
import random
import re, pathlib
import pickle
from scipy.stats import norm
import math
import torch
import numpy as np
from abc import ABC
from datetime import datetime
from ftfy import fix_text
import os
from tqdm import tqdm
from pathlib import Path
import sys
from itertools import combinations
import argparse
from trueskill import Rating, rate_1vs1
import adaptive_utils
import jsonlines
import beir_eval
import trueskill
import json
from listwise_reranking_modules import ListwiseLLM

logger = logging.getLogger(__name__)


class Runner():
    def __init__(self, args):
        self.args = args
        self.dataset, self.output = self.load_data(args)
        self.method = args.method
        self.llm = ListwiseLLM(self.args)
        self.num_calls = 0
        self.total_calls = 0
        self.ill_format = False
        # set seed
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        torch.cuda.manual_seed_all(0)
        self.winsize = 0
        self.trueskill_time = 0

    def load_data(self, args):
        path = f'./data/{args.firststage}/{args.dataset}.jsonl'
        data = []
        output_data = []
        with jsonlines.open(path, 'r') as reader:
            for x in reader:
                data.append(x)
        if os.path.exists(self.args.output_path) and self.args.resume:
            with jsonlines.open(path, 'r') as reader:
                for x in reader:
                    output_data.append(x)
            output_len = len(output_data)
            if output_len > 0:
                assert data[output_len-1]['qid'] == output_data[0]['qid']
            data = data[output_len:]
            print(f"Resuming from idx: {len(output_data)}..")
        print(f'Cutting to top {self.args.top_k}')
        for x in data:
            x['bm25_results'] = x['bm25_results'][:self.args.top_k]
        return data, output_data
    
    def write_jsonl_file(self, results):
        Path(self.args.output_path).parent.mkdir(exist_ok=True, parents=True)
        with jsonlines.open(self.args.output_path, 'w') as writer:
            writer.write_all(results)
        print(f"Writing to {self.args.output_path} done!")
    
    def restore_missing_and_remove_additional_indices(self, response, max_response):
        # remove additional indices
        orig_response = response
        if len(response) == 0:
            return list(range(max_response+1))
        if max(response) != max_response: # if include larger index, hallucination: remove that
            response = [x for x in response if x <= max_response]
        if len(response) == 0: # inserted break instead of assertion
            print(f"Unwanted scnario 1: orig_response was {orig_response}, max response {max_response}, cleaned up response: {response} (0). Filling in with set(range(max_response+1)).")
            response = list(range(max_response + 1))
        if max(response) != max_response:
            print(f"Unwanted scenario: model output max {max(response)}, but the real max response is {max_response}.\nResponse: {response}\n")
        full_indices = set(range(max_response + 1))
        given_indices = set(response)
        missing_indices = sorted(full_indices - given_indices)
        restored_list = response + missing_indices
        return restored_list
    # input: request class (query+candidates), start idx, end idx
    # output: permutation for start idx ~ end_idx, in list form
    # output ex: [1, 2, 0, 3, 6, 12, 17, 19, 4, ... 13, 16] (total len 20)
    def run_listwise_reranking(self, query, list_of_candidates, verbose=True):
        if len(list_of_candidates) == 0:
            return []
        if len(list_of_candidates) == 1:
            return [0]
        self.winsize += len(list_of_candidates)
        if self.args.debug_for_numpass:
            response = list(range(len(list_of_candidates)))
            self.num_calls += 1
            self.total_calls += 1
            return response
        prompt, in_token_count = self.llm.create_prompt(query, list_of_candidates)
        # permutation: plain string, raw output from llm
        permutation, out_token_count = self.llm.run_llm(prompt, len(list_of_candidates))
        # clean permutation, remove duplicates
        response = self.llm._clean_response(permutation)
        response = [int(x) - 1 for x in response.split() if int(x) > 0] # only leave out positive values
        response = self.llm._remove_duplicate(response)
        if sorted(response) != list(range(len(list_of_candidates))):
            self.ill_format = True
            #print(f"Ill-formatted output: {permutation}\nCleaned up to {[x+1 for x in response]}")
        else:
            self.ill_format = False
        max_response = len(list_of_candidates) - 1
        response = self.restore_missing_and_remove_additional_indices(response, max_response)
        # check if output is a permutation of given index range
        if sorted(response) != list(range(len(list_of_candidates))):
            print(f"Wrong permutation output on run_listwise_reranking!!")
            import pdb; pdb.set_trace()
        if self.total_calls % self.args.print_freq == 0 and verbose:
            print(f"Example I/O for total calls: {self.total_calls}, instance-wise num calls: {self.num_calls}:")
            if type(prompt) == list:
                print(prompt[1]['content'][:1000] + prompt[1]['content'][-2000:])
            else:
                print(prompt[:1000]+ '\n...\n' + prompt[-2000:])
            print(f"\nOutput: {permutation}\nCleaned up to: {[x+1 for x in response]}")
        self.num_calls += 1
        self.total_calls += 1
        return response

    def run_sliding_windows(self, instance):
        candidates = instance['bm25_results']
        n = len(candidates)
        stride = self.args.window_size // 2
        start_indices = list(range(max(0, n - self.args.window_size), -1, -stride))
        if n > 1 and 0 not in start_indices:
            start_indices.append(0) # ensure to always include 0 on start_indices even if n is not dividable by stride
        progress_log = ''
        for start_idx in start_indices:
            end_idx = min(start_idx + self.args.window_size, n)   # 범위 넘치지 않게
            cut_range = copy.deepcopy(candidates[start_idx:end_idx])
            progress_log += f", {start_idx} - {end_idx}"
            response = self.run_listwise_reranking(instance['q_text'], candidates[start_idx:end_idx])
            # Extract the relevant candidates and create a mapping for new order
            original_rank = [tt for tt in range(len(cut_range))]
            response = [ss for ss in response if ss in original_rank]
            response = response + [tt for tt in original_rank if tt not in response]
            # Update candidates in the new order
            for j, x in enumerate(response):
                candidates[j + start_idx] = copy.deepcopy(cut_range[x])
                if candidates[j + start_idx]['bm25_score'] is not None:
                    candidates[j + start_idx]['bm25_score'] = cut_range[j]['bm25_score']
        instance['bm25_results'] = candidates
        if instance.get('num_calls') == None:
            instance['num_calls'] = self.num_calls
        else:
            instance['num_calls'] += self.num_calls
        self.num_calls = 0
        return instance

    def update_ratings(self, orderings, window_candidates):
        # exception handling: bypass if len(orderings) == 1
        if len(orderings) <= 1:
            return window_candidates
            import pdb; pdb.set_trace()
        else:
            assert len(orderings) == len(window_candidates) # should not happen if we get correct run_listwise_reranking output
            sorted_candidates = [window_candidates[i] for i in orderings]
            try:
                start = time.time()
                updated_ratings = trueskill.rate([[x['rating']] for x in sorted_candidates],
                    ranks=range(len(sorted_candidates)))
                end = time.time()
                self.trueskill_time += (end-start)
            except Exception as e:
                print(f"Error: {e}")
                import pdb; pdb.set_trace()
            for i, x in enumerate(sorted_candidates):
                x['rating'] = updated_ratings[i][0]
            # restore initial orderings
            inverse_order = [0] * len(orderings)
            for new_idx, old_idx in enumerate(orderings):
                try:
                    inverse_order[old_idx] = new_idx
                except Exception as e:
                    print(f"Error: {e}")
                    import pdb; pdb.set_trace()
            window_candidates = [sorted_candidates[inverse_order[i]]
                         for i in range(len(orderings))]
            return window_candidates

    def run_fixed_budget(self, instance):
        num_candidates = len(instance['bm25_results'])
        if num_candidates == 0:
            instance['num_calls'] = 0
            print("Num_candidates was 0! Skipping run_trueskill")
            return instance
        instance['initial_pass'] = 0
        # initialize candidates with Trueskill Rating
        for x in instance['bm25_results']:
            x['rating'] = Rating() # initialize with mu=25, sigma=8.33
        if self.args.use_firststage_orderings:
            print(f"Using firststage orderings")
            for i, x in enumerate(instance['bm25_results']):
                x['rating'] = Rating(mu=x['bm25_score'], sigma=x['bm25_score']/3)
        query = instance['q_text']
        iteration = 0
        # skip computation if length is 0 or 1
        if num_candidates < 2:
            instance['num_calls'] = 0
            return instance
        candidates = instance['bm25_results']
        total_stage = len(self.args.budget_per_stage)
        for stage_idx, iter_for_stage in enumerate(self.args.budget_per_stage):
            iter_for_stage = int(iter_for_stage)
            subset_candidates = copy.deepcopy(candidates[:self.args.window_size*iter_for_stage])
            chosen_indices = list(range(len(candidates)))[:self.args.window_size*iter_for_stage]
            grouped_candidates, grouped_candidxs = self.group_candidates(subset_candidates)
            # for print
            group_print = ''
            start_idx = chosen_indices[0]
            for y in grouped_candidxs:
                x = [a+start_idx for a in y]
                group_print += f"[{str(x[:2])[1:-1]} ... {str(x[-2:])[1:-1]}], "
            iteration += len(grouped_candidates)
            for chunked_candidates, chunked_candidxs in zip(grouped_candidates, grouped_candidxs):
                if len(chunked_candidates) == 0:
                    print(f"Length of chunked candidates is 0! Should not happen")
                    import pdb; pdb.set_trace()
                ordering = self.run_listwise_reranking(query, chunked_candidates)
                updated_subset = self.update_ratings(ordering, chunked_candidates)
                # update trueskill values on candidates
                for i in range(len(updated_subset)):
                    orig_idx = chosen_indices[chunked_candidxs[i]]
                    if candidates[orig_idx]['pid'] != updated_subset[i]['pid']:
                        import pdb; pdb.set_trace()
                    candidates[orig_idx] = updated_subset[i]
            candidates.sort(key=lambda cand: cand['rating'].mu, reverse=True)
        instance['bm25_results'] = candidates
        # replace score with trueskill mu
        for x in instance['bm25_results']:
            # Convert Rating to JSON
            x['rating'] = {'mu': x['rating'].mu, 'sigma': x['rating'].sigma}
            # update scores to mu values
            x['bm25_score'] = x['rating']['mu']
        instance['num_calls'] = iteration
        return instance

    def run_trueskill(self, instance):
        num_candidates = len(instance['bm25_results'])
        if num_candidates == 0:
            print("Num_candidates was 0! Skipping run_trueskill")
            instance['num_calls'] = 0
            return instance
        # initialize candidates with Trueskill Rating
        if self.args.use_firststage_orderings:
            print(f"Using firststage orderings")
            if self.args.options == 'independent_firststage':
                n = len(instance['bm25_results'])
                max_gap, mu = 0.49, 25
                #s=mu*sigma_ratio
                a = max_gap/max(n-1,1);m=(n-1)/2
                mu_scores = [mu+a*(m-r) for r in range(n)]
                for i, x in enumerate(instance['bm25_results']):
                    x['rating'] = Rating(mu=mu_scores[i], sigma=mu_scores[i]/3)
            else:
                for i, x in enumerate(instance['bm25_results']):
                    x['rating'] = Rating(mu=x['bm25_score'], sigma=x['bm25_score']/3)
        else:
            for x in instance['bm25_results']:
                x['rating'] = Rating() # initialize with mu=25, sigma=8.33
        initial_cnt = 0
        query = instance['q_text']
        # step 1: Iterate over candidate list in windows of size m
        for start in range(0, num_candidates, self.args.window_size):
            end = min(start + self.args.window_size, num_candidates)
            orderings = self.run_listwise_reranking(query, instance['bm25_results'][start:end])
            initial_cnt += 1
            # Create a safe copy of the current window of candidates
            window_candidates = copy.deepcopy(instance['bm25_results'][start:end])
            # Update TrueSkill ratings for this window's candidates using listwise reranking
            window_candidates = self.update_ratings(orderings, window_candidates)
            # (This function should update each candidate.rating based on the window's current ordering.)
            # Sort the copied window candidates by descending TrueSkill mu (higher skill first)
            # for debug
            #window_mu = [window_candidates[x].rating.mu for x in orderings]
            #window_sigma = [window_candidates[x].rating.sigma for x in orderings]
            window_candidates.sort(key=lambda cand: cand['rating'].mu, reverse=True)
            # Assign the sorted candidates back into the result in the correct window segment
            instance['bm25_results'][start:end] = window_candidates
        # global sort over all candidates
        instance['bm25_results'].sort(key=lambda cand: cand['rating'].mu, reverse=True)
        instance['initial_pass'] = initial_cnt
        instance['bm25_results'] = self._adaptive_prob_rank(instance)
        # replace score with trueskill mu
        for x in instance['bm25_results']:
            # Convert Rating to JSON
            x['rating'] = {'mu': x['rating'].mu, 'sigma': x['rating'].sigma}
            # update scores to mu values
            x['bm25_score'] = x['rating']['mu']
        instance['num_calls'] = self.num_calls
        self.num_calls = 0
        return instance

    def group_candidates(self, subset_candidates):
        total = len(subset_candidates)
        if self.args.chunking_mode in ['sequential', 'random']:
            temp = list(range(total))
            if self.args.chunking_mode == 'random':
                random.shuffle(temp)
            chunked_orderings = [temp[i:i + self.args.window_size] for i in range(0, len(subset_candidates), self.args.window_size)]
            chunked_candidates = [[subset_candidates[x] for x in y] for y in chunked_orderings]
            return chunked_candidates, chunked_orderings

    def _get_topR_candidate_pids(self, candidates):
        sorted_candidates = sorted(candidates, key=lambda cand: cand['rating'].mu, reverse=True)[:self.args.R]
        return [x['pid'] for x in sorted_candidates]

    def _adaptive_prob_rank(self, instance):
        if self.args.break_mode == 'top10_nochange':
            return self._adaptive_prob_rank_top10_nochange(instance)
        elif self.args.break_mode == 'reduce_uncertain':
            return self._adaptive_prob_rank_reduce_uncertain(instance)

    def _adaptive_prob_rank_reduce_uncertain(self, instance):
        num_candidates = len(instance['bm25_results'])
        candidates = instance['bm25_results']
        query = instance['q_text']
        iteration = 0
        # skip computation if length is 0 or 1
        if num_candidates < 2:
            return candidates
        should_break = False
        while True: # iterate, update candidates
            mus = torch.tensor([x['rating'].mu for x in candidates])
            sigmas = torch.tensor([x['rating'].sigma for x in candidates])
            threshold_sumto = self.args.R / num_candidates
            t = adaptive_utils._find_threshold_binary_search(threshold_sumto, mus, sigmas)
            probs_above_t = np.array(self._compute_probs_above_t(mus, sigmas, t))
            tol = self.args.tol
            # re-order candidates by their probs_above_t values
            sorted_pairs = sorted(zip(candidates, probs_above_t), key=lambda x: x[1], reverse=True)
            sorted_candidates = [cand for cand, _ in sorted_pairs]
            candidates = sorted_candidates
            sorted_probs_above_t = np.array([s_i for _, s_i in sorted_pairs])
            if 'filter_only_nonrelevant' in self.args.options:
                mask = (sorted_probs_above_t > tol)
            else:
                mask = (sorted_probs_above_t > tol) & (sorted_probs_above_t < (1.0 - tol))
            chosen_indices = np.where(mask)[0]
            if len(chosen_indices) < self.args.uncertain_U: # run final iteration, break after final rerank
                uncertain_len = len(chosen_indices)
                chosen_indices = np.where(sorted_probs_above_t > tol)[0]
                should_break = True
            if self.args.use_fixed_window_size:
                need_to_add = 0
                if len(chosen_indices) % self.args.window_size != 0:
                    need_to_add = self.args.window_size - len(chosen_indices) % self.args.window_size
                # add higher rank first, and add lower rank last
                while need_to_add > 0:
                    if chosen_indices[0] > 0:
                        chosen_indices = np.append(chosen_indices[0] - 1, chosen_indices)
                        need_to_add -= 1
                    elif chosen_indices[-1] < (len(candidates) - 1):
                        chosen_indices = np.append(chosen_indices, chosen_indices[-1] + 1)
                        need_to_add -= 1
                    else:
                        break
                if len(chosen_indices) % self.args.window_size != 0 and len(candidates) != 100:
                    print(f"Orig chosen_indices: {np.where(mask)[0]}\nnew: {chosen_indices} len {len(chosen_indices)}")
                    print("Error: Not dividable by window size!!")
                    import pdb; pdb.set_trace()
            subset_candidates = [copy.deepcopy(candidates[i]) for i in chosen_indices]
            grouped_candidates, grouped_candidxs = self.group_candidates(subset_candidates)
            if iteration + len(grouped_candidates) >= self.args.hard_constraint:
                additional = self.args.hard_constraint - iteration
                grouped_candidates = grouped_candidates[:additional]
                grouped_candidxs = grouped_candidxs[:additional]
            iteration += len(grouped_candidates)
            instance['levels'].append(len(grouped_candidates))
            for chunked_candidates, chunked_candidxs in zip(grouped_candidates, grouped_candidxs):
                if len(chunked_candidates) == 0:
                    print(f"Length of chunked candidates is 0! Should not happen")
                    import pdb; pdb.set_trace()
                ordering = self.run_listwise_reranking(query, chunked_candidates)
                updated_subset = self.update_ratings(ordering, chunked_candidates)
                # update trueskill values on candidates
                for i in range(len(updated_subset)):
                    orig_idx = chosen_indices[chunked_candidxs[i]]
                    if candidates[orig_idx]['pid'] != updated_subset[i]['pid']:
                        import pdb; pdb.set_trace()
                    candidates[orig_idx] = updated_subset[i]
            if iteration >= self.args.hard_constraint or should_break:
                candidates.sort(key=lambda cand: cand['rating'].mu, reverse=True)
                if iteration >= self.args.hard_constraint:
                    print(f"@@@@@@@@@@ Forced break after {self.args.hard_constraint}th iteration @@@@@@@@@")
                else:
                    print(f"Stop since length of uncertain passages {uncertain_len} was smaller than R={self.args.R}, breaking after running final indices of {len(chosen_indices)}")
                break
        return candidates

    def _adaptive_prob_rank_top10_nochange(self, instance):
        num_candidates = len(instance['bm25_results'])
        candidates = instance['bm25_results']
        query = instance['q_text']
        iteration = 0
        nochange_cnt = 0
        # skip computation if length is 0 or 1
        if num_candidates < 2:
            return candidates
        prev_topR_pids = self._get_topR_candidate_pids(candidates)
        while True: # iterate, update candidates
            mus = torch.tensor([x['rating'].mu for x in candidates])
            sigmas = torch.tensor([x['rating'].sigma for x in candidates])
            threshold_sumto = self.args.R / num_candidates
            t = adaptive_utils._find_threshold_binary_search(threshold_sumto, mus, sigmas)
            probs_above_t = np.array(self._compute_probs_above_t(mus, sigmas, t))
            tol = self.args.tol
            # re-order candidates by their probs_above_t values
            sorted_pairs = sorted(zip(candidates, probs_above_t), key=lambda x: x[1], reverse=True)
            sorted_candidates = [cand for cand, _ in sorted_pairs]
            candidates = sorted_candidates
            sorted_probs_above_t = np.array([s_i for _, s_i in sorted_pairs])
            mask = (sorted_probs_above_t > tol) & (sorted_probs_above_t < (1.0 - tol))
            chosen_indices = np.where(mask)[0]
            subset_candidates = [copy.deepcopy(candidates[i]) for i in chosen_indices]
            grouped_candidates, grouped_candidxs = self.group_candidates(subset_candidates)
            iteration += len(grouped_candidates)
            for chunked_candidates, chunked_candidxs in zip(grouped_candidates, grouped_candidxs):
                if len(chunked_candidates) == 0:
                    print(f"Length of chunked candidates is 0! Should not happen")
                    import pdb; pdb.set_trace()
                ordering = self.run_listwise_reranking(query, chunked_candidates)
                updated_subset = self.update_ratings(ordering, chunked_candidates)
                # update trueskill values on candidates
                for i in range(len(updated_subset)):
                    orig_idx = chosen_indices[chunked_candidxs[i]]
                    if candidates[orig_idx]['pid'] != updated_subset[i]['pid']:
                        import pdb; pdb.set_trace()
                    candidates[orig_idx] = updated_subset[i]
            cur_topR_pids = self._get_topR_candidate_pids(candidates)
            if self.total_calls % self.args.print_freq == 0:
                print(f"Prev pid:   {prev_topR_pids}\n=> Cur pid: {cur_topR_pids}")
            # if top-R orderings didn't change after last iteration, break.
            if prev_topR_pids == cur_topR_pids:
                nochange_cnt += 1
            break_option = nochange_cnt >= self.args.nochange_cnt and not self.ill_format
            if iteration >= self.args.hard_constraint or break_option:
                candidates.sort(key=lambda cand: cand['rating'].mu, reverse=True)
                if iteration >= self.args.hard_constraint:
                    print(f"@@@@@@@@@@ Forced break after {self.args.hard_constraint}th iteration @@@@@@@@@")
                else:
                    print("Stop since top-10 pids doesn't change")
                break
            prev_topR_pids = cur_topR_pids
        return candidates

    def _compute_probs_above_t(self, mus, sigmas, t):
        s = []
        for mu_i, sigma_i in zip(mus, sigmas):
            z = (t - mu_i) / sigma_i
            s.append(float(1.0 - norm.cdf(z)))  # P(X_i > t)
        return s

    def run_reranking(self):
        full_data = self.dataset
        for pass_iter in range(self.args.num_pass):
            print(f"Total iteration: {pass_iter+1}/{self.args.num_pass}")
            results = copy.deepcopy(self.output)
            for idx, instance in tqdm(enumerate(full_data), total=len(full_data)):
                instance['levels'] = []
                if self.args.method == 'sliding_windows':
                    result = self.run_sliding_windows(instance)
                elif self.args.method == 'trueskill':
                    result = self.run_trueskill(instance)
                elif self.args.method == 'fixed_budget':
                    result = self.run_fixed_budget(instance)
                else:
                    raise NotImplementedError(f"Method {args.method} not implemented")
                results.append(result)
            full_data = copy.deepcopy(results)
            ndcg_10, out_string = beir_eval.run_rerank_eval(results, combined=True)
            print(f"Mid process, pass_iter={pass_iter}: {ndcg_10}")
            print(f"Avg winsize: {self.winsize}/{self.total_calls} ({self.winsize / self.total_calls})")
        num_calls = np.array([x['num_calls'] for x in results])
        avgcnt = np.mean(num_calls).item()
        print(f"\nAverage reranking count: {avgcnt}\n\n")
        print(f"\nMax reranking count: {np.max(num_calls).item()}\n\n")
        if not self.args.debug_for_numpass:
            self.write_jsonl_file(results)
            print(f"Writing jsonl to {self.args.output_path} done, for full length!")
        try:
            ndcg_10, out_string = beir_eval.run_rerank_eval(results, combined=True)
        except:
            print('Error happened during running run_rerank_eval. skipping evaluation')
            ndcg_10, out_string = 'None', 'None'
        print(f"\nNum_calls: {num_calls}")
        return_values = {'eval_results': out_string, 'ndcg_10': ndcg_10,
                'args': str(vars(self.args)),
                'avg_reranking_cnt': avgcnt}
        return return_values

def build_path(args, root="outputs", skip={"resume", "print_freq", "onlyeval", "top_k", "api", 'level_analysis', 'debug_for_numpass'}):
    clean = lambda x: re.sub(r"[^\w\-.]+", "_", str(x))
    parts = []
    for k, v in sorted(vars(args).items()):
        if k in skip: continue
        if isinstance(v, bool): v = int(v)          # 1 / 0
        elif isinstance(v, (list, tuple)): v = "-".join(map(str, v))
        parts.append(f"{clean(k)}={clean(v)}")
    return str(pathlib.Path(root, *parts))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dl19', type=str)
    parser.add_argument('--method', default='sliding_windows', type=str) # sliding_windows, trueskill, fixed_budget
    parser.add_argument('--chunking_mode', default='sequential', type=str) # random, sequential
    parser.add_argument('--break_mode', default='top10_nochange', type=str) # top10_nochange, reduce_uncertain
    parser.add_argument('--tol', default=1e-8, type=float)
    parser.add_argument('--window_size', default=20, type=int)
    parser.add_argument('--top_k', default=1000, type=int)
    parser.add_argument('--debug_for_numpass',  action='store_true')
    parser.add_argument('--use_fixed_window_size', action='store_true')
    parser.add_argument('--budget_per_stage', type=int, metavar='INT', nargs='+', default=[5, 1, 1, 3]) # only use for fixed_budget
    parser.add_argument('--nochange_cnt', default=1, type=int) # only use for top10_nochange, when to break: bigger the more resource
    parser.add_argument('--print_freq', default=100, type=int) # frequency to print examples
    parser.add_argument('--model_path', default='castorini/rank_zephyr_7b_v1_full', type=str)
    parser.add_argument('--R', default=10, type=int) # only for trueskill
    parser.add_argument('--hard_constraint', default=100, type=int) # only for trueskill
    parser.add_argument('--uncertain_U', default=10, type=int) # only for trueskill
    parser.add_argument('--num_pass', default=1, type=int)
    parser.add_argument('--firststage', default='bm25', type=str)
    parser.add_argument('--use_firststage_orderings', action='store_true')
    parser.add_argument('--options', default='', type=str)
    parser.add_argument('--onlyeval', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--level_analysis', action='store_true')
    parser.add_argument('--api', default='sy', type=str) # sy, jy


    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    # configure output path
    model_name = args.model_path.replace('/','_')
    output_path = build_path(args, root='./results/')
    args.output_path = output_path
    print(f"Output path: {output_path}")

    if args.onlyeval:
        if not os.path.exists(args.output_path):
            print(f"Output path {args.output_path} does not exist!!")
            raise Exception
        else:
            data = []
            with jsonlines.open(output_path, 'r') as reader:
                for x in reader:
                    data.append(x)

            # only for analysis
            analysis = []
            for x in data:
                ndcg_10, _ = beir_eval.run_rerank_eval([x], combined=True)
                analysis.append((x['num_calls'], ndcg_10))
            print(f"\n\nAnalysis (num_calls, ndcg):\n{analysis}\n\n")
            num_calls = [x[0] for x in analysis]
            avg_num_calls = np.array(num_calls).mean()
            max_num_calls = max(num_calls)
            print(f"Avg num calls: {avg_num_calls}, max: {max_num_calls}")
            full_ndcg_10, _ = beir_eval.run_rerank_eval(data, combined=True)
            if args.level_analysis:
                levels = [x['levels'] for x in data]
                print(f"Levels: ")
                for i, x in enumerate(levels):
                    print(f"({i}): {x}")
                import pdb; pdb.set_trace()
    else:
        runner = Runner(args)
        runner.run_reranking()
        if args.profile:
            runner.llm.prof.stop_profile()
            flops = runner.llm.prof.get_total_flops()
            print(f"Flops: {flops}")
        print(f"Trueskill time: {runner.trueskill_time}")
