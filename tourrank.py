import random
import copy
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
from run import Runner
from argparse import Namespace
import beir_eval
import jsonlines

def sort_docs_by_relevance(doc_ids, relevance_scores):

    combined = list(zip(doc_ids, relevance_scores))

    sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
    
    sorted_doc_ids = [doc_id for doc_id, score in sorted_combined]
    
    return sorted_doc_ids

def dcg_at_k(scores, k):

    scores = np.asfarray(scores)[:k]
    if scores.size:
        return np.sum(scores / np.log2(np.arange(2, scores.size + 2)))
    
    return 0.0

def ndcg_at_k(scores, k):
    dcg_max = dcg_at_k(sorted(scores, reverse=True), k)
    if not dcg_max:
        return 0.
    
    return dcg_at_k(scores, k) / dcg_max

def get_top_M(answer, N=10, M=5, groups_docid=[]):

    # temp = answer.split('\n\n')[-1]
    temp = answer.split('\n')
    temp_length = len(temp)
    for i in range(1, temp_length+1):
        # print(i*(-1),temp_length)
        if 'Document' in temp[i*(-1)]:
            temp = temp[i*(-1)]
            break
    temp = temp.split(':')[-1]
    temp = temp.split('.')[0]
    temp = temp.split(',')
    top_M = []
    for doc in temp:
        try:
            try:
                flag = 0
                if '...' in doc:
                    flag = 1
                if flag == 0:
                    doc_num = int(doc.split()[-1]) - 1
                    top_M.append(doc_num)
            except IndexError:
                print('IndexError occured in doc. (get_top_M), just ignore it.')
                print(doc)
                # debug
                s = 'IndexError, ' + str(N) + ', ' + str(M) + ', ' + answer
                debug_path = './debug.txt'
                with open(debug_path, 'a') as f:
                    f.write('New Error: ' + '\n')
                    f.write(str(top_M) + '\n')
                    f.write(doc + '\n')
                    f.write(s + '\n')
                    f.write('end' + '\n' + '\n')

        except ValueError:
            for j in range(1, N+1):
                if j not in top_M:
                    doc_num = j
                    # top_M.append(j)
                    break
            print('ValueError occured in score. (get_top_M)')
            top_M.append(doc_num)
            # debug
            s = 'ValueError, ' + str(N) + ', ' + str(M) + ', ' + answer
            debug_path = './debug.txt'
            with open(debug_path, 'a') as f:
                f.write('New Error: ' + '\n')
                f.write(str(top_M) + '\n')
                f.write(doc + '\n')
                f.write(s + '\n')
                f.write('end' + '\n' + '\n')
    
    top_M_ids = []
    for doc_num in top_M:
        top_M_ids.append(groups_docid[doc_num])

    return top_M_ids

def get_final_top(answer, groups_docid):
    temp = answer.split('>')
    ranked_list = []
    for doc in temp:
        try:
            try:
                doc_num = int(doc.split()[-1]) - 1
                ranked_list.append(doc_num)
            except IndexError:
                print('IndexError occured in doc. (get_final_top), just ignore it.')
        except ValueError:
            for j in range(1, N+1):
                if j not in ranked_list:
                    # ranked_list.append(j)
                    doc_num = j
                    break
            print('ValueError occured in score. (get_final_top)')
            ranked_list.append(doc_num)
    
    ranked_docids_list = []
    for doc_num in ranked_list:
        ranked_docids_list.append(groups_docid[doc_num])

    return ranked_docids_list

def get_groups_chunk(docs_id, N=10):

    doc_num = len(docs_id)
    docs_groups = []
    cur_num = 0
    while cur_num < doc_num:
        docs_groups.append(docs_id[cur_num: cur_num + N])
        cur_num += N
    
    return docs_groups

def get_groups_skip(docs_id, to_n_groups=10, m_docs_per_group=10):

    # doc_num = len(docs_id)
    docs_groups = []
    for i in range(to_n_groups):
        cur_group = []
        for j in range(m_docs_per_group):
            idx = j * to_n_groups + i
            if idx < len(docs_id):
                cur_group.append(docs_id[idx])
            #cur_group.append(docs_id[j*to_n_groups + i])
        docs_groups.append(cur_group)
    
    return docs_groups

def group_processing(groups, query, N, M, all_contents, runner, return_top = True):

    group_score_dict = {}

    random.shuffle(groups)

    Tour_condadidates = []

    for doc in groups:
        single_condadidate = {}
        single_condadidate['text'] = all_contents[doc]
        single_condadidate['title'] = ""
        Tour_condadidates.append(single_condadidate)

    response = runner.run_listwise_reranking(query, Tour_condadidates, verbose=False)[:M]

    response = [doc_no + 1 for doc_no in response]


    def format_doc_list(doc_indices):
        return ', '.join([f'Document {i}' for i in doc_indices])

    answer = format_doc_list(response)

    top_M_ids = get_top_M(answer, N=N, M=M, groups_docid=groups)
    for doc_id in top_M_ids:
        group_score_dict[doc_id] = 1

    if return_top:
        return top_M_ids
    else:
        return group_score_dict



def filter_processing(y_it, query, docs_id, all_contents, runner):

    total_pass = 0

    # initialize the score of each doc_id (global)
    docs_score_dict = {}
    for doc in docs_id:
        docs_score_dict[doc] = 0


    #########################################################
    ################## stage 1  start #######################
    #########################################################

    if len(docs_id) > 50:

        # 100->50->20->10->5; Y times
        # 100->50, 5 groups
        N=20
        M=10
        # initialize the score of each doc_id (this stage)
        groups_score_dict_list = []


        stage1_docs_id = docs_id
        docs_groups = get_groups_skip(stage1_docs_id, to_n_groups=5, m_docs_per_group=N)

        for y in range(len(docs_groups)):
            groups = docs_groups[y]
            group_score_dict = group_processing(groups, query, N, M, all_contents, runner, False)
            total_pass += 1
            groups_score_dict_list.append(group_score_dict)

        # combine the results
        for group_score_dict in groups_score_dict_list:
            for doc, score in group_score_dict.items():
                docs_score_dict[doc] += score

        # get randed list
        ranked_list = sort_docs_by_relevance(list(docs_score_dict.keys()), list(docs_score_dict.values()))

    else :
        ranked_list = docs_id

    #########################################################
    ################## stage 1  end   #######################
    #########################################################


    #########################################################
    ################## stage 2  start #######################
    #########################################################

    if len(docs_id) > 20:
        # 50->20, 5 groups
        N=10
        M=4
        stage2_docs_id = ranked_list[: 50]

        # initialize the score of each doc_id (this stage)
        groups_score_dict_list = []
        stage2_docs_id = ranked_list[: 50]
        docs_groups = get_groups_skip(stage2_docs_id, to_n_groups=5, m_docs_per_group=N)

        for y in range(len(docs_groups)):
            groups = docs_groups[y]
            group_score_dict = group_processing(groups, query, N, M, all_contents, runner, False)
            total_pass += 1
            groups_score_dict_list.append(group_score_dict)

        for group_score_dict in groups_score_dict_list:
            for doc, score in group_score_dict.items():
                docs_score_dict[doc] += score

        ranked_list = sort_docs_by_relevance(list(docs_score_dict.keys()), list(docs_score_dict.values()))

    else :
        ranked_list = docs_id

    #########################################################
    ################## stage 2  end   #######################
    #########################################################


    #########################################################
    ################## stage 3  start #######################
    #########################################################


    if len(docs_id) > 10:
        # 20->10;
        N=20
        M=10
        stage3_docs_id = ranked_list[: 20]
        docs_groups = get_groups_skip(stage3_docs_id, to_n_groups=1, m_docs_per_group=N)
        for bat in range(len(docs_groups)):
            groups = docs_groups[bat]

            #################single processing#################
            top_M_ids = group_processing(groups, query, N, M, all_contents, runner, True)
            total_pass += 1
            #################single processing#################

            for doc_id in top_M_ids:
                docs_score_dict[doc_id] += 1

        ranked_list = sort_docs_by_relevance(list(docs_score_dict.keys()), list(docs_score_dict.values()))

    else :
        ranked_list = docs_id

    #########################################################
    ################## stage 3  end   #######################
    #########################################################


    #########################################################
    ################## stage 4  start #######################
    #########################################################

    if len(docs_id) > 5:
        # 10->5;
        N=10
        M=5
        stage4_docs_id = ranked_list[: 10]
        docs_groups = get_groups_skip(stage4_docs_id, to_n_groups=1, m_docs_per_group=N)
        for bat in range(len(docs_groups)):
            groups = docs_groups[bat]

            #################single processing#################
            top_M_ids = group_processing(groups, query, N, M, all_contents, runner, True)
            total_pass += 1
            #################single processing#################

            for doc_id in top_M_ids:
                docs_score_dict[doc_id] += 1

        ranked_list = sort_docs_by_relevance(list(docs_score_dict.keys()), list(docs_score_dict.values()))

    else :
        ranked_list = docs_id

    #########################################################
    ################## stage 4  end   #######################
    #########################################################


    #########################################################
    ################## stage 5  start #######################
    #########################################################


    if len(docs_id) > 2:
        # 5->2;
        N=5
        M=2
        stage4_docs_id = ranked_list[: 5]
        docs_groups = get_groups_skip(stage4_docs_id, to_n_groups=1, m_docs_per_group=N)
        for bat in range(len(docs_groups)):
            groups = docs_groups[bat]

            #################single processing#################
            top_M_ids = group_processing(groups, query, N, M, all_contents, runner, True)
            total_pass += 1
            #################single processing#################

            for doc_id in top_M_ids:
                docs_score_dict[doc_id] += 1
    else:
        ranked_list = docs_id

    #########################################################
    ################## stage 5  end   #######################
    #########################################################

    return docs_score_dict, total_pass


def convert_dataset(example):
    """
    Convert one example from the source format to the desired format.
    """
    converted = {
        'query': example['q_text'],
        'hits': []
    }

    for rank, doc in enumerate(example['bm25_results'], start=1):
        hit = {
            'content': doc['text'],
            'qid': example['qid'],
            'docid': doc['pid'],
            'rank': rank,
            'score': doc['bm25_score']
        }
        converted['hits'].append(hit)

    return converted


def convert_qrels_to_trec_str(qid, qrels_dict):
    """
    Convert a qrels dictionary to TREC-formatted string with newline separation.

    Args:
        qid (int or str): Query ID
        qrels_dict (dict): Dictionary where key=docid, value=relevance

    Returns:
        str: Multiline string in format: qid 0 docid relevance
    """
    lines = []
    for docid, relevance in qrels_dict.items():
        lines.append(f"{qid} 0 {docid} {relevance}")
    return "\n".join(lines)


def merge_multiple_qrels(qrels_list):
    return "\n".join([convert_qrels_to_trec_str(qid, qrels) for qid, qrels in qrels_list])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dl19', type=str)
    parser.add_argument('--rep', default=10, type=int)
    parser.add_argument('--model_path', default='castorini/rank_zephyr_7b_v1_full', type=str)

    args = parser.parse_args()

    print(f"handling dataset: {args.dataset}")
    print(f"handling rep: {args.rep}")

    # calling runner 

    args = Namespace(
        rep=args.rep,
        dataset= args.dataset,
        method='sliding_windows', 
        chunking_mode='sequential', 
        break_mode='top10_nochange',
        tol=1e-8,
        window_size=20,
        use_fixed_window_size=False,
        budget_per_stage=[5, 1, 1, 3],  
        nochange_cnt=1,
        print_freq=10000,
        model_path=args.model_path,
        R=10, 
        hard_constraint=100,  
        uncertain_U=10, 
        num_pass=1,
        firststage='bm25',
        top_k=100,
        debug_for_numpass=False,
        use_firststage_orderings=False,
        options='',
        onlyeval=False,
        output_path= './output'
    )

    runner = Runner(args)

    query_docs = []

    for instance in runner.dataset:
        query_docs.append(convert_dataset(instance))
    all_queries = [item['query'] for item in query_docs]


    # get the ideal ranking permutation
    all_docs, all_contents, = [], {}
    for i in range(len(query_docs)):
        cur = query_docs[i]
        query = cur['query']
        docs = cur['hits']
        cur_docs = []
        for doc in docs:
            qid = doc['qid']
            doc_id = doc['docid']
            content = doc['content']

            cur_docs.append(doc_id)
            all_contents[doc_id] = content

        all_docs.append(cur_docs)

    result_list = []

    total_pass_list = []


    for index in tqdm(range(len(all_queries)), desc="Processing Queries"):
        total_pass_sum = 0
        if index <= -1:
            continue
        query = all_queries[index]
        docs_id = copy.deepcopy(all_docs[index])

        # initialize the score of each doc
        docs_score_dict = {}
        for doc in docs_id:
            docs_score_dict[doc] = 0


        docs_score_dicts_list = []
        processes = []
        Y = args.rep
        for y in range(Y):
            docs_score_dict, total_pass_single = filter_processing(y, query, docs_id, all_contents, runner)
            docs_score_dicts_list.append(docs_score_dict)
            total_pass_sum += total_pass_single

        # initialize the global score of each doc
        global_docs_score_dict = {}
        for doc in docs_id:
            global_docs_score_dict[doc] = 0

        for y in range(len(docs_score_dicts_list)):
            cur_docs_score_dict = docs_score_dicts_list[y]
            for doc_id, score in cur_docs_score_dict.items():
                global_docs_score_dict[doc_id] += score
            # get ranked list
            ranked_list = sort_docs_by_relevance(list(global_docs_score_dict.keys()), list(global_docs_score_dict.values()))


        ranked_list = sort_docs_by_relevance(list(global_docs_score_dict.keys()), list(global_docs_score_dict.values()))
        result = copy.deepcopy(runner.dataset[index])
        for j in range(len(result['bm25_results'])):
            result['bm25_results'][j]['pid'] = ranked_list[j]
        
        total_pass_list.append(total_pass_sum)

        result_list.append(result)

    mean_total_pass = sum(total_pass_list) / len(total_pass_list)

    print("beir_eval result:")

    ndcg_10, out_string = beir_eval.run_rerank_eval(result_list, combined=True)

    print("")
    print("TourRank original NDCG results")
    print("")
    print(f"mean_total_pass: {mean_total_pass}")

    # write jsonl file
    output_path = f"tourrank_results/{args.model_path}/first-stage-{args.firststage}-top{args.top_k}/{args.dataset}/rep_{args.rep}_pass_{mean_total_pass}_{ndcg_10}.jsonl"
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    with jsonlines.open(output_path, 'w') as writer:
        writer.write_all(result_list)
    print(f"Write to: {output_path} done!")

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")  

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    file_name = f"{args.dataset}_{timestamp}.txt"
    file_path = os.path.join(log_dir, file_name)

    result_str = (
        "TourRank original NDCG results\n"
        f"mean_total_pass: {mean_total_pass:.1f}\n"
        "\n"
        "\n"
        f"UACRank beir_eval result:\n{out_string}\n"
    )

    with open(file_path, "w") as f:
        f.write(result_str)

    print(f"Evaluation results saved to {file_path}")
