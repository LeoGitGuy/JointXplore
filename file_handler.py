import json
import pickle

def load_annotations_questions_filenames(data_path, filename_prefix, num_samples):
    with open(f"{data_path}{filename_prefix}filtered_questions.json") as f:
        questions = json.load(f)[:num_samples]
    print(f"Number of Questions: {len(questions)}\nExample Question: {questions[0]}")
    with open(f"{data_path}{filename_prefix}filtered_annotations.json") as f:
        annotations = json.load(f)[:num_samples]
    print(f"Number of Annotations: {len(annotations)}\nExample Annotation: {annotations[0]}")
    with open(f"{data_path}{filename_prefix}image_filenames.json") as f:
        file_names = json.load(f)
    print(f"Number of filenames: {len(file_names)}\nExample Filename: {file_names[0]}")
    return questions, annotations, file_names

def load_raw_annotations_questions(data_path):
    with open(f"{data_path}v2_OpenEnded_mscoco_train2014_questions.json") as f:
        train_questions = json.load(f)
    with open(f"{data_path}v2_mscoco_train2014_annotations.json") as f:
        train_annotations = json.load(f)
    with open(f"{data_path}v2_OpenEnded_mscoco_val2014_questions.json") as f:
        questions = json.load(f)   
    with open(f"{data_path}v2_mscoco_val2014_annotations.json") as f:
        annotations = json.load(f)
    with open(f"{data_path}train_image_filenames.json") as f:
        train_file_names = json.load(f)
    with open(f"{data_path}image_filenames.json") as f:
        file_names = json.load(f)
    return train_questions, train_annotations, questions, annotations, train_file_names, file_names

def save_coverage_results(model_type, task, num_samples, s, activations, coverage_dict):
    with open(f"./coverage_results/{model_type}_{task}_{num_samples}_{s}", "wb") as fp:
        pickle.dump(activations, fp)
    if coverage_dict != {}:
        with open(f"./coverage_results/{model_type}_{task}_{num_samples}_{s}.json", "w") as fp:
            json.dump(coverage_dict, fp)

def save_attack_results(model_type, task, num_attacks, s, attacks_results_summary, attack_results):
    with open(f"./adversarial_results/{model_type}_{task}_{num_attacks}_{s}.json", "w") as fp:
        json.dump(attacks_results_summary, fp)
    with open(f"./adversarial_results/{model_type}_{task}_{num_attacks}_{s}", "wb") as fp:
        pickle.dump(attack_results, fp)
        
def save_filtered_datasets(filtered_questions, filtered_annotations, data_path, filename_prefix=""):
    with open(f"{data_path}{filename_prefix}filtered_questions", "w") as fp:
        json.dump(filtered_questions, fp)
    with open(f"{data_path}{filename_prefix}filtered_annotations", "w") as fp:
        json.dump(filtered_annotations, fp)