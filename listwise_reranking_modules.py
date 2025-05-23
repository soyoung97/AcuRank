import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from ftfy import fix_text
import torch
from transformers.generation import GenerationConfig
import openai
from openai import OpenAI
import tiktoken

ALPH_START_IDX = ord('A') - 1

class ListwiseLLM():
    def __init__(self, args):
        self.args = args
        self._system_message = 'You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query.'
        self.use_openai = args.model_path.lower() in {'o4-mini', 'gpt-4o', 'o3-mini', 'gpt-4o-mini', 'gpt-4.1-mini'}
        if self.use_openai:
            self.api_key = "YOUR_API_KEY_HERE"
            print(f"Using Openai API KEY of {self.args.api}!")
            self.model = OpenAI(api_key=self.api_key)
            self.tokenizer = tiktoken.encoding_for_model('gpt-4o') # self.args.model_path)
            self.model_max_tokens = 4096
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.args.model_path, device_map='auto', 
                                                          torch_dtype=torch.bfloat16)
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)
                                                          #attn_implementation='flash_attention_2', 
            self.num_max_output_tokens = len(self.tokenizer.encode(" > ".join([f"[{i+1}]" for i in range(self.args.window_size)]))) - 1
            self.model_max_tokens = 4096
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({"pad_token": "<|end_of_text|>"})

    def _replace_number(self, s):
        return re.sub(r"\[(\d+)\]", r"(\1)", s)
    
    def convert_doc_to_prompt_content(self, doc, max_length):
        content = doc['text']
        if "title" in doc and doc["title"]:
            content = "Title: " + doc["title"] + " " + "Content: " + content
        content = content.strip()
        content = fix_text(content)
        content = " ".join(content.split()[:int(max_length)])
        return self._replace_number(content)
    
    def get_num_tokens(self, prompt):
        return len(self.tokenizer.encode(prompt))
    
    def _clean_response(self, response):
        new_response = ""
        for c in response:
            if not c.isdigit():
                new_response += " "
            else:
                new_response += c
        new_response = new_response.strip()
        return new_response

    def _remove_duplicate(self, response):
        new_response = []
        for c in response:
            if c not in new_response:
                new_response.append(c)
        return new_response

    def num_output_tokens(self, candidate_length):
        if candidate_length <= 20 and self.args.model_path == 'castorini/rank_zephyr_7b_v1_full':
            cached_mapping = {20:90, 19:85, 18:80, 17:75, 16:70, 15:65, 14:60, 13:55, 12:50, 11:45, 10:40,
                    9:35, 8:31, 7:27, 6:23, 5:19, 4:15, 3:11, 2:7, 1:3, 0:0}
            return cached_mapping[candidate_length]
        else:
            token_str = ' > '.join([f"[{i+1}]" for i in range(candidate_length)])
            output_estimate = len(self.tokenizer.encode(token_str)) - 1
            return output_estimate

    # list of {'text', 'title', 'bm25_score', 'pid'}
    def create_prompt(self, query, list_of_candidates):
        num = len(list_of_candidates)
        if num > 20:
            print(f"Exception: got list num over 20: {num}. This is unexpected")
            import pdb; pdb.set_trace()
        query = self._replace_number(query)
        max_length = 300 * (20 / (num))
        orig_max_length = max_length
        iteration = 0
        while True:
            rank = 0
            user_text = f"I will provide you with {num} passages, each indicated by  a numerical identifier []. Rank the passages based on their relevance to the search query: {query}.\n\n"
            for cand in list_of_candidates:
                rank += 1
                content = self.convert_doc_to_prompt_content(cand, max_length)
                user_text += f"[{rank}] {content}\n"
            user_text += f"Search Query: {query}.\nRank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., [2] > [1], Only respond with the ranking results, do not say any word or explain."
            messages = [{"role": "system", "content": self._system_message},
                        {"role": "user", "content": user_text}]
            if self.use_openai:
                num_tokens = self.get_num_tokens(self._system_message + "\n" + user_text)
            else:
                prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True)
                prompt = fix_text(prompt)
                num_tokens = self.get_num_tokens(prompt)
            if num_tokens <= (self.model_max_tokens - self.num_output_tokens(num)):
                break
            else:
                max_length_before = max_length
                max_length -= max(
                        1,
                        (
                            num_tokens
                            - self.model_max_tokens
                            + self.num_output_tokens(num)
                        )
                        // (num * 4),
                    )
                if max_length < 0:
                    if iteration < (orig_max_length + 1):
                        print(f"Max length for prompt resulted in {max_length}: Reverting it to {max_length_before} -> {max_length_before-1}")
                        max_length = max_length_before - 1 #  to avoid scenarios where initial length was too long and max length suddenly dropped to minus
                    else: # iteration > 10000 or max_length_before <= 1:
                        print(f"MaxLen: {max_length}, max_length_before: {max_length_before}, iteration: {iteration}. Seems like the query and passage itself is longer than model max tokens!!! (on listwise_reranking_modules) just breaking..")
                        import pdb; pdb.set_trace()
                        break
        if self.use_openai:
            return messages, num_tokens
        else:
            return prompt, num_tokens
    
    def run_llm(self, prompt, candidate_number):
        if self.use_openai:
            # ChatGPT API
            response = self.model.chat.completions.create(
                model=self.args.model_path,
                messages=prompt,  # = messages list
                #max_completion_tokens=self.num_output_tokens(candidate_number) + 20,
            )
            outputs = response.choices[0].message.content.strip()
            out_len = len(self.tokenizer.encode(outputs))
            return outputs, out_len

        inputs = self.tokenizer([prompt])
        inputs = {k: torch.tensor(v).to('cuda') for k, v in inputs.items()}
        gen_cfg = GenerationConfig.from_model_config(self.model.config)
        gen_cfg.max_new_tokens = self.num_output_tokens(candidate_number) #self.num_max_output_tokens
        gen_cfg.min_new_tokens = self.num_output_tokens(candidate_number) #self.num_max_output_tokens
        # gen_cfg.temperature = 0
        gen_cfg.do_sample = False
        output_ids = self.model.generate(**inputs, generation_config=gen_cfg)
        if self.model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
        outputs = self.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
        )
        #print(f"Input len: {inputs['input_ids'].shape[1]}, Output: {outputs}")
        return outputs, output_ids.size(0)
