import torch, gc
num_samples = 4
import textattack
import copy
from textattack.models.wrappers import ModelWrapper
from textattack.attack_recipes import Seq2SickCheng2018BlackBox, PWWSRen2019
import json
from tqdm import tqdm

#from textattack.attack_recipes import PWWSRen2019, Seq2SickCheng2018BlackBox
from textattack import Attacker
from textattack.metrics.attack_metrics import (
    AttackQueries,
    AttackSuccessRate,
    WordsPerturbed,
)
from torch import nn
import re
from torch.utils.data import DataLoader
from load_helper import VQADataset, VQADataset_random_img, collate_fn, VQADataset_Albef, VQADataset_Albef_random_img, collate_fn_albef

def generate_attack_dataset(questions, annotations, model_type):
    data = []
    for idx, q in enumerate(questions):
        if annotations[idx]['labels'] != []:
            score_idx = torch.tensor(annotations[idx]['scores']).argmax().item()
            label = annotations[idx]["labels"][score_idx]
            data.append((q["question"], label))
        else:
            # no answer, 545 corresponds to unknown
            if model_type == "vilt":
                data.append((q["question"], 545))
            elif model_type == "albef":
                data.append((q["question"], "unkown"))
    return textattack.datasets.Dataset(data)


class ViltWrapper(ModelWrapper):
    def __init__(self, model, questions, annotations, processor, config, id_to_filename, use_rnd, answer_list = []):
            self.model = model
            self.questions = questions
            self.annotations = annotations
            self.processor = processor
            self.config = config
            self.id_to_filename = id_to_filename
            self.use_rnd = use_rnd

    def __call__(self, text_input_list):
        num_samples = len(text_input_list)
        new_questions = []
        new_annotations = []

        for i in range(num_samples):
            new_questions.append(copy.deepcopy(self.questions[0]))
            new_questions[i]["question"] = text_input_list[i]
            new_annotations.append(self.annotations[0])
        if self.use_rnd:
            dataset = VQADataset_random_img(questions=new_questions,
                        annotations=new_annotations,
                        processor=self.processor,
                        config = self.config,
                        id_to_filename=self.id_to_filename)
        else:
            dataset = VQADataset(questions=new_questions,
                        annotations=new_annotations,
                        processor=self.processor,
                        config = self.config,
                        id_to_filename=self.id_to_filename)
        loader = DataLoader(dataset, collate_fn=collate_fn, batch_size=num_samples, shuffle=False)
        with torch.no_grad():
            batch = next(iter(loader))
            outputs = self.model(**batch)
            sm = outputs.logits.double()
            sm = torch.nn.functional.log_softmax(sm, dim=1)
            sm = torch.exp(sm)
        return sm
    
class AlbefWrapper(ModelWrapper):
    def __init__(self, model, questions, annotations, processor, config, id_to_filename, use_rnd, answer_list = []):
            self.model = model
            self.questions = questions
            self.annotations = annotations
            self.vis_processor = processor[0]
            self.txt_processor = processor[1]
            self.config = config
            self.id_to_filename = id_to_filename
            self.answer_list = answer_list
            self.use_rnd = use_rnd

    def __call__(self, text_input_list):
        num_samples = len(text_input_list)
        new_questions = []
        new_annotations = []
        #new_questions = copy.deepcopy(questions[0])
        # for idx, nq in enumerate(new_questions):
        #   nq["question"] = text_input_list[idx]
        for i in range(num_samples):
            new_questions.append(copy.deepcopy(self.questions[0]))
            new_questions[i]["question"] = text_input_list[i]
            new_annotations.append(self.annotations[0])
        if self.use_rnd:
            dataset = VQADataset_Albef_random_img(questions=new_questions,
                        annotations=new_annotations,
                        vis_processor=self.vis_processor,
                        txt_processor=self.txt_processor,
                        config = self.config,
                        id_to_filename=self.id_to_filename)
        else:
            dataset = VQADataset_Albef(questions=new_questions,
                            annotations=new_annotations,
                            vis_processor=self.vis_processor,
                            txt_processor=self.txt_processor,
                            config = self.config,
                            id_to_filename=self.id_to_filename)
        loader = DataLoader(dataset, collate_fn=collate_fn_albef, batch_size=num_samples, shuffle=False)
        with torch.no_grad():
            batch = next(iter(loader))
            preds = self.model.predict_answers(samples = {"image": batch["image"], "text_input": batch["text_input"]},
                                      answer_list = self.answer_list)
        return preds
    

def execute_attack(model, questions, annotations, processor, config, id_to_filename, model_type, num_attacks, data_path, use_rnd):
    with open(f"{data_path}/answer_list.json") as f:
        answer_list = json.load(f)
    attack_results = []
    attack_results_dict = []
    def construct_attacker(i):
        attack_args = textattack.AttackArgs(
        num_examples=1,
        disable_stdout=True,
        silent=True,
        query_budget=30
        )
        if model_type == "vilt":
            model_wrapper = ViltWrapper(model, questions[i:i+1], annotations[i:i+1], processor, config, id_to_filename, use_rnd)
            attack = PWWSRen2019.build(model_wrapper)
        elif model_type == "albef":
            model_wrapper = AlbefWrapper(model, questions[i:i+1], annotations[i:i+1], processor, config, id_to_filename, use_rnd, answer_list)
            attack = Seq2SickCheng2018BlackBox.build(model_wrapper)
        attacker = Attacker(attack, attack_dataset, attack_args=attack_args)
        return attacker
    i = 0
    j = 0
    while j < num_attacks:
        gc.collect()
        torch.cuda.empty_cache()
        attack_dataset = generate_attack_dataset(questions[i:i+1], annotations[i:i+1], model_type)
        attacker = construct_attacker(i)
        attack_result = attacker.attack_dataset()
        attack_result, attack_dict, skipped = process_attack_result(attack_result[0], answer_list, model_type)
        attack_results.append(attack_result)
        attack_results_dict.append(attack_dict)
        if not skipped:
            j+=1
        i+=1
    summary = generate_attack_summary(attack_results)
    attacks_results_summary = {"attack_results": attack_results_dict, "summary": summary}
    return attack_results, attacks_results_summary
    

def process_attack_result(attack_result, answer_list, model_type):
    new_result = {}
    skipped = isinstance(attack_result, textattack.attack_results.skipped_attack_result.SkippedAttackResult)
    if model_type == "vilt" and isinstance(attack_result, textattack.attack_results.successful_attack_result.SuccessfulAttackResult):
        matches = re.findall("\d+", attack_result.goal_function_result_str())
        original = int(matches[0])
        if original != 0:
            original = original-1
        adversarial = int(matches[2])
        if adversarial != 0:
            adversarial = adversarial-1
        new_result["change_word"] = f"{answer_list[original]} ({matches[1]}%) --> {answer_list[adversarial]} ({matches[3]}%)"
    new_result["change_id"] = attack_result.goal_function_result_str()
    new_result["original_text"] = attack_result.original_text()
    new_result["perturbed_text"] = attack_result.perturbed_text()
    
    return attack_result, new_result, skipped

def generate_attack_summary(attack_results):
    attack_success_stats = AttackSuccessRate().calculate(attack_results)
    words_perturbed_stats = WordsPerturbed().calculate(attack_results)
    attack_query_stats = AttackQueries().calculate(attack_results)
    summary = dict(attack_success_stats, **words_perturbed_stats)
    summary.update(attack_query_stats)
    summary.pop('num_words_changed_until_success', None)
    return summary
