import json
import argparse
import torch
import pickle
from torch.utils.data import DataLoader
from transformers import ViltConfig, ViltProcessor, ViltForQuestionAnswering

from attach_hooks import get_all_activation_layers, get_activations, read_activations, Modes, compute_boundary_strong_coverage, compute_ksection_coverage
from load_helper import filename_mapping, annotations_preprocessing, collate_fn, collate_fn_albef, get_dataset_and_model
from eval_helper import eval_model
#from prepare_attack import construct_attacker, process_attack_result, generate_attack_summary


def parse_args():
    parser = argparse.ArgumentParser(description="JointXplore")

    parser.add_argument("--data-path", help="path to filtered vqa 2.0 dataset", default="./data/")
    parser.add_argument("--model", help="specify the model to use", choices=["vilt", "albef"], default="vilt")
    parser.add_argument("--use_rnd_images", help="if specified, random images are used", action="store_true")
    parser.add_argument("--task", required = True, help="choose task to execute", choices=["coverage_regions", "coverage", "adversarial_text"])
    parser.add_argument("--num_samples", help="number of dataset samples for experiment", type=int, default=2500)
    parser.add_argument("--activations_file", help="for K-Section or Boundary Coverage, reference coverage regions must be loaded")
    parser.add_argument("--batch_size", help="specify batch size", default=4)
    parser.add_argument("--num_sections", help="Number of sections for K-Section Boundary", type=int, default=10)
    parser.add_argument("--num_attacks", help="Number of target for adversarial text attack", type=int, default=1)
    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def main():
    print("Starting")
    args = parse_args()
    filename_prefix = ""
    num_attacks = args.num_attacks
    task = args.task
    model_type = args.model
    data_path = args.data_path
    num_samples = args.num_samples
    image_path = f"{data_path}val2014"
    use_rnd = args.use_rnd_images
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(num_samples)
    print(type(num_samples))
    # if args.num_samples is not None:
    #     num_samples = args.num_samples
    # else:
    # check which data to load
    if task == "coverage_regions":
        filename_prefix = "train_"
        image_path = f"{data_path}train2014"
        #num_samples = 5000
        activations = {}
        modes = [Modes.REGION]
    elif task == "coverage":
        try:
            activations = read_activations(args.activations_file)
            modes = [Modes.KSECTION, Modes.BOUNDARY]
        except:
            raise("Please specify path to activations file as --actviations_file \
                \nFor K-Section and Boundary coverage a reference coverage must be provided.")
    # load data    
    with open(f"{data_path}{filename_prefix}filtered_questions.json") as f:
        questions = json.load(f)[:num_samples]
    print(f"Number of Questions: {len(questions)}\nExample Question: {questions[0]}")
    with open(f"{data_path}{filename_prefix}filtered_annotations.json") as f:
        annotations = json.load(f)[:num_samples]
    print(f"Number of Annotations: {len(annotations)}\nExample Annotation: {annotations[0]}")
    with open(f"{data_path}{filename_prefix}image_filenames.json") as f:
        file_names = json.load(f)
    print(f"Number of filenames: {len(file_names)}\nExample Filename: {file_names[0]}")
    
    # build dicts to get filename from question id and id from filename
    filename_to_id, id_to_filename = filename_mapping(file_names=file_names,root=image_path)
    # create Vilt config
    config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    # preprocess annotations to get weighted scores and labels
    annotations = annotations_preprocessing(config, annotations)
    # get dataset and model
    dataset, model, processor, answer_list = get_dataset_and_model(model_type, config, id_to_filename, device, questions, annotations, use_rnd, data_path)   
    # put model on gpu if available
    model = model.to(device)
    if use_rnd:
        s = "rnd_images"
    else:
        s = "full_images"
    if "coverage" in task:
        # attach hooks for coverage mode which was specified in inputs arguments
        get_all_activation_layers(model, modes, model_type, k=args.num_sections)
        # create dataloader from dataset
        if model_type == "vilt":
            dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=False)
        elif model_type == "albef":
            dataloader = DataLoader(dataset, collate_fn = collate_fn_albef, batch_size=args.batch_size, shuffle=False)
        # run evaluation
        total_loss, total_acc = eval_model(model, dataloader, device, model_type, answer_list)
        if model_type == "vilt":
            print(f"Avg loss on pretrained model with {s}: {total_loss}")
        print(f"Total Acc on pretrained model with {s}: {total_acc}")
        
        if task == "coverage":
            activations = get_activations()
            print(activations.keys())
            activations = compute_ksection_coverage(activations)
            activations = compute_boundary_strong_coverage(activations)

        with open(f"./coverage_results/{model_type}_{task}_{num_samples}_{s}", "w") as fp:
            json.dump(activations, fp)
    elif task == "adversarial_text":
        attack_results = []
        attack_results_dict = []
        with open(f"{data_path}/answer_list.json") as f:
            answer_list = json.load(f)
        for i in range(num_attacks):
            attacker = construct_attacker(model, questions[i:i+1], annotations[i:i+1], processor, config, id_to_filename)
            attacker.update_attack_args(num_examples = 1)
            attack_result = attacker.attack_dataset()
            attack_result, attack_dict = process_attack_result(attack_result[0], answer_list)
            attack_results.append(attack_result)
            attack_results_dict.append(attack_dict)
        summary = generate_attack_summary(attack_results)
        attacks_results_summary = {"attack_results": attack_results_dict, "summary": summary}
        with open(f"./adversarial_results/{model_type}_{task}_{num_attacks}_{s}.json", "w") as fp:
            json.dump(attacks_results_summary, fp)
        with open(f"./adversarial_results/{model_type}_{task}_{num_attacks}_{s}", "wb") as fp:
            pickle.dump(attack_results, fp)
        
    print("Finished")

if __name__ == "__main__":
    main()

    

    
    
    
    
    