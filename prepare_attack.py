import torch
num_samples = 4
import textattack
import copy
from textattack.models.wrappers import ModelWrapper
from textattack.attack_recipes import PWWSRen2019
from textattack import Attacker
from textattack.metrics.attack_metrics import (
    AttackQueries,
    AttackSuccessRate,
    WordsPerturbed,
)
from torch import nn
import re
from torch.utils.data import DataLoader
from load_helper import VQADataset, VQADataset_random_img, collate_fn

def generate_attack_dataset(questions, annotations):
  data = []

  for idx, q in enumerate(questions):
    if annotations[idx]['labels'] != []:
      score_idx = torch.tensor(annotations[idx]['scores']).argmax().item()
      label = annotations[idx]["labels"][score_idx]
      data.append((q["question"], label))
    else:
      # no answer, 545 corresponds to unknown
      data.append((q["question"], 545))
  return textattack.datasets.Dataset(data)


class ViltWrapper(ModelWrapper):
    def __init__(self, model, questions, annotations, processor, config, id_to_filename,):
            self.model = model
            self.questions = questions
            self.annotations = annotations
            self.processor = processor
            self.config = config
            self.id_to_filename = id_to_filename

    def __call__(self, text_input_list):
        print(f"Text Input List: {text_input_list}")
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
            #print(f"Output sum: {(sm.sum(dim=1)-1).abs()}")
        return sm
    
def construct_attacker(model, questions, annotations, processor, config, id_to_filename,):
    attack_dataset = generate_attack_dataset(questions, annotations)
    model_wrapper = ViltWrapper(model, questions, annotations, processor, config, id_to_filename,)
    attack = PWWSRen2019.build(model_wrapper)
    attacker = Attacker(attack, attack_dataset)
    return attacker

def process_attack_result(attack_result, answer_list):
    new_result = {}
    if isinstance(attack_result, textattack.attack_results.successful_attack_result.SuccessfulAttackResult):
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
    
    return attack_result, new_result

def generate_attack_summary(attack_results):
    attack_success_stats = AttackSuccessRate().calculate(attack_results)
    words_perturbed_stats = WordsPerturbed().calculate(attack_results)
    attack_query_stats = AttackQueries().calculate(attack_results)
    summary = dict(attack_success_stats, **words_perturbed_stats)
    summary.update(attack_query_stats)
    summary.pop('num_words_changed_until_success', None)
    return summary
