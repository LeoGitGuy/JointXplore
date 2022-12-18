import json
import argparse
import torch
import pickle
from torch.utils.data import DataLoader
from transformers import ViltConfig
from tqdm import tqdm
from attach_hooks import get_all_activation_layers, get_activations, read_activations, Modes, compute_boundary_strong_coverage, compute_ksection_coverage
from load_helper import filename_mapping, annotations_preprocessing, collate_fn, collate_fn_albef, get_dataset_and_model
from eval_helper import eval_model
from prepare_attack import execute_attack
from file_handler import load_annotations_questions_filenames, save_coverage_results, save_attack_results


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

    if task == "coverage_regions":
        filename_prefix = "train_"
        image_path = f"{data_path}train2014"
        activations = read_activations(args.activations_file)
        modes = [Modes.REGION]
    elif task == "coverage":
        activations = read_activations(args.activations_file)
        modes = [Modes.KSECTION, Modes.BOUNDARY]
        if activations == {}:
            raise("Please specify path to activations file as --actviations_file \
                \nFor K-Section and Boundary coverage a reference coverage must be provided.")
    elif task == "adversarial_text":
        num_samples = 1000
    # load data
    questions, annotations, file_names = load_annotations_questions_filenames(data_path, filename_prefix, num_samples)
    # build dicts to get filename from question id and id from filename
    filename_to_id, id_to_filename = filename_mapping(file_names=file_names,root=image_path)
    # create Vilt config
    config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    # preprocess annotations to get weighted scores and labels
    annotations = annotations_preprocessing(config, annotations, model_type)
    # get dataset and model
    dataset, model, answer_list, processor = get_dataset_and_model(model_type, config, id_to_filename, device, questions, annotations, use_rnd, data_path)   
    # put model on gpu if available
    model = model.to(device)
    if use_rnd:
        s = "rnd_images"
    else:
        s = "full_images"
    if "coverage" in task:
        # attach hooks for coverage mode which was specified in inputs arguments
        get_all_activation_layers(model, modes, model_type, k=args.num_sections, batch_size=args.batch_size)
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
        coverage_dict = {}
        if task == "coverage":
            activations = get_activations()
            activations, coverage_dict = compute_ksection_coverage(activations)
            activations, coverage_dict = compute_boundary_strong_coverage(activations)
            coverage_dict["Accuracy"] = total_acc
        save_coverage_results(model_type, task, num_samples, s, activations, coverage_dict)
    elif task == "adversarial_text":
        attack_results, attacks_results_summary = execute_attack(model, questions, annotations, processor, config, id_to_filename, model_type, num_attacks, data_path, use_rnd)
        save_attack_results(model_type, task, num_attacks, s, attacks_results_summary, attack_results)
        
    print("Finished")

if __name__ == "__main__":
    main()

    

    
    
    
    
    