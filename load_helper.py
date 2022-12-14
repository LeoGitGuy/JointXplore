import os
import torch
from PIL import Image
import numpy as np
import imagesize
import re
import json
from typing import Optional
from tqdm.notebook import tqdm
from torch.nn.utils.rnn import pad_sequence
from transformers import ViltProcessor, ViltForQuestionAnswering
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = ""

def id_from_filename(filename: str) -> Optional[int]:
	filename_re = re.compile(r".*(\d{12})\.((jpg)|(png))")
	match = filename_re.fullmatch(filename)
	if match is None:
		return None
	return int(match.group(1))

def get_score(count: int) -> float:
	return min(1.0, count / 3)

def filename_mapping(file_names, root):
	filename_to_id = {root + "/" + file: id_from_filename(file) for file in file_names}
	id_to_filename = {v:k for k,v in filename_to_id.items()}   
	return filename_to_id, id_to_filename

def annotations_preprocessing(config, annotations):
	for annotation in tqdm(annotations):
		answers = annotation['answers']
		answer_count = {}
		for answer in answers:
			answer_ = answer["answer"]
			answer_count[answer_] = answer_count.get(answer_, 0) + 1
		labels = []
		scores = []
		for answer in answer_count:
			if answer not in list(config.label2id.keys()):
				continue
			labels.append(config.label2id[answer])
			score = get_score(answer_count[answer])
			scores.append(score)
		annotation['labels'] = labels
		annotation['scores'] = scores
	return annotations

def get_dataset_and_model(model_type, config, id_to_filename, device, questions, annotations, use_rnd, data_path):
    answer_list = []
    global processor
    if model_type == "vilt":
        processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        # load dataset
        if not use_rnd:
            dataset = VQADataset(questions=questions,
                        annotations=annotations,
                        processor=processor,
                        config = config,
                        id_to_filename= id_to_filename)
        else:
            dataset = VQADataset_random_img(questions=questions,
                        annotations=annotations,
                        processor=processor,
                        config = config,
                        id_to_filename= id_to_filename)
        # load model
        model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa",
                                                    num_labels=len(config.id2label),
                                                    id2label=config.id2label,
                                                    label2id=config.label2id)
    elif model_type == "albef":
        with open(f"{data_path}/answer_list.json") as f:
            answer_list = json.load(f)
        #try: 
        os.chdir("./LAVIS")
        from lavis.models import load_model_and_preprocess
        model, vis_processors, processor = load_model_and_preprocess(name="albef_vqa", model_type="vqav2", is_eval=True, device=device)
        os.chdir("../")
        #except:
        #    raise("Import error")
        if not use_rnd:
            dataset = VQADataset_Albef(questions=questions,
                        annotations=annotations,
                        vis_processor = vis_processors, 
                        txt_processor = processor,
                        config = config,
                        id_to_filename = id_to_filename)
        else:
            dataset = VQADataset_Albef_random_img(questions=questions,
                        annotations=annotations,
                        vis_processor = vis_processors, 
                        txt_processor = processor,
                        config = config,
                        id_to_filename = id_to_filename)
    return dataset, model, processor, answer_list


### For Vilt Model
class VQADataset(torch.utils.data.Dataset):
	"""VQA (v2) dataset."""

	def __init__(self, questions, annotations, processor, config, id_to_filename):
		# takes in questions, annotations and processor
		self.questions = questions
		self.annotations = annotations
		self.processor = processor
		self.config = config
		self.id_to_filename = id_to_filename

	def __len__(self):
		return len(self.annotations)

	def __getitem__(self, idx):
		# get image + text
		annotation = self.annotations[idx]
		questions = self.questions[idx]
		image = Image.open(self.id_to_filename[annotation['image_id']])
		text = questions['question']
		# encode image and text
		encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
		# remove batch dimension
		for k,v in encoding.items():
			encoding[k] = v.squeeze()
		# add labels
		labels = annotation['labels']
		scores = annotation['scores']
		# based on: https://github.com/dandelin/ViLT/blob/762fd3975c180db6fc88f577cf39549983fa373a/vilt/modules/objectives.py#L301
		# create soft encoding vectors for labels based on the labels and scores
		targets = torch.zeros(len(self.config.id2label))
		for label, score in zip(labels, scores):
				targets[label] = score
		encoding["labels"] = targets
		encoding["label_indices"] = labels

		return encoding
	
# Dataset with random images
class VQADataset_random_img(torch.utils.data.Dataset):
	"""VQA (v2) dataset."""

	def __init__(self, questions, annotations, processor, config, id_to_filename):
		# takes in questions, annotations and processor
		self.questions = questions
		self.annotations = annotations
		self.processor = processor
		self.config = config
		self.id_to_filename = id_to_filename

	def __len__(self):
		return len(self.annotations)

	def __getitem__(self, idx):
		# get image + text
		annotation = self.annotations[idx]
		questions = self.questions[idx]
		#width, height = imagesize.get(id_to_filename[annotation['image_id']])
		#image = Image.open(id_to_filename[annotation['image_id']])
		arr = np.random.randint(
				low=0, 
				high=256,
				size=(640, 478, 3),
				dtype=np.uint8
			)
		image = Image.fromarray(arr)

		text = questions['question']
		# encode image and text
		encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
		# remove batch dimension
		for k,v in encoding.items():
			encoding[k] = v.squeeze()
		# add labels
		labels = annotation['labels']
		scores = annotation['scores']
		# based on: https://github.com/dandelin/ViLT/blob/762fd3975c180db6fc88f577cf39549983fa373a/vilt/modules/objectives.py#L301
		# create soft encoding vectors for labels based on the labels and scores
		targets = torch.zeros(len(self.config.id2label))
		for label, score in zip(labels, scores):
				targets[label] = score
		encoding["labels"] = targets
		encoding["label_indices"] = labels

		return encoding
	
def filter_dataset_for_greyscale(annotations, questions, id_to_filename, data_len = 10000):
  	# delete_questions = []
	filtered_annotations = []
	filtered_questions = []
	for i in tqdm(range(len(annotations))):
		try:
			filename = id_to_filename[annotations[i]['image_id']]
			image = Image.open(filename)
			if not len(image.getbands()) == 1:
				filtered_annotations.append(annotations[i])
				filtered_questions.append(questions[i])
			if len(filtered_annotations) == data_len:
				break
			elif len(filtered_annotations) % 500 == 0:
				print(len(filtered_annotations))
			else:
				print(f"Deleted datapoint {i}")
		except:
			continue
  #sorted_annotations = [annotations[idx] for idx in tqdm(range(len(annotations))) if not len(Image.open(id_to_filename[annotations[idx]['image_id']]).getbands()) == 1]
  
	return filtered_annotations, filtered_questions


# Create a batch, for each batch we need to pad the images because they are not always same size
def collate_fn(batch):
	input_ids = [item['input_ids'] for item in batch]
	pixel_values = [item['pixel_values'] for item in batch]
	attention_mask = [item['attention_mask'] for item in batch]
	token_type_ids = [item['token_type_ids'] for item in batch]
	labels = [item['labels'] for item in batch]
	label_indices = [torch.tensor(item['label_indices']) for item in batch]
	
	# create padded pixel values and corresponding pixel mask
	encoding = processor.feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
	padded_label_indices = pad_sequence(label_indices, batch_first = True, padding_value = -1)
	# create new batch
	batch = {}
	batch['input_ids'] = torch.stack(input_ids)
	batch['attention_mask'] = torch.stack(attention_mask)
	batch['token_type_ids'] = torch.stack(token_type_ids)
	batch['pixel_values'] = encoding['pixel_values']
	batch['pixel_mask'] = encoding['pixel_mask']
	batch['labels'] = torch.stack(labels)
	#batch['label_indices'] = padded_label_indices
	
	return batch

def collate_fn_train(batch):
	input_ids = [item['input_ids'] for item in batch]
	pixel_values = [item['pixel_values'] for item in batch]
	attention_mask = [item['attention_mask'] for item in batch]
	token_type_ids = [item['token_type_ids'] for item in batch]
	labels = [item['labels'] for item in batch]
	#label_indices = [torch.tensor(item['label_indices']) for item in batch]

	# create padded pixel values and corresponding pixel mask
	encoding = processor.feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
	#padded_label_indices = pad_sequence(label_indices, batch_first = True, padding_value = -1)
	# create new batch
	batch = {}
	batch['input_ids'] = torch.stack(input_ids)
	batch['attention_mask'] = torch.stack(attention_mask)
	batch['token_type_ids'] = torch.stack(token_type_ids)
	batch['pixel_values'] = encoding['pixel_values']
	batch['pixel_mask'] = encoding['pixel_mask']
	batch['labels'] = torch.stack(labels)
	#batch['label_indices'] = padded_label_indices

	return batch

### For Albef Model

class VQADataset_Albef(torch.utils.data.Dataset):
	"""VQA (v2) dataset."""

	def __init__(self, questions, annotations, vis_processor, txt_processor, config, id_to_filename):
		# takes in questions, annotations and processor
		self.questions = questions
		self.annotations = annotations
		self.vis_processors = vis_processor
		self.txt_processors = txt_processor
		self.config = config
		self.id_to_filename = id_to_filename

	def __len__(self):
		return len(self.annotations)

	def __getitem__(self, idx):
		# get image + text
		annotation = self.annotations[idx]
		questions = self.questions[idx]
		image = Image.open(self.id_to_filename[annotation['image_id']])
		text = questions['question']
		# encode image and text
		encoding = {}
		encoding["image"] = self.vis_processors["eval"](image).to(device)
		encoding["question"] =  self.txt_processors["eval"](text)
		# add labels
		encoding['answers'] = annotation['labels']
		encoding['scores'] = annotation['scores']
		#encoding["label_indices"] = labels

		return encoding

class VQADataset_Albef_random_img(torch.utils.data.Dataset):
	"""VQA (v2) dataset."""

	def __init__(self, questions, annotations, vis_processor, txt_processor, config, id_to_filename):
		# takes in questions, annotations and processor
		self.questions = questions
		self.annotations = annotations
		self.vis_processors = vis_processor
		self.txt_processors = txt_processor
		self.config = config
		self.id_to_filename = id_to_filename
        
	def __len__(self):
		return len(self.annotations)

	def __getitem__(self, idx):
		# get image + text
		annotation = self.annotations[idx]
		questions = self.questions[idx]
		#width, height = imagesize.get(id_to_filename[annotation['image_id']])
		#image = Image.open(id_to_filename[annotation['image_id']])
		arr = np.random.randint(
			  low=0, 
			  high=256,
			  size=(640, 478, 3),
			  dtype=np.uint8
		  )
		image = Image.fromarray(arr)
		
		text = questions['question']
		# encode image and text
		encoding = {}
		encoding["image"] = self.vis_processors["eval"](image).to(device)
		encoding["question"] =  self.txt_processors["eval"](text)
		# add labels
		# based on: https://github.com/dandelin/ViLT/blob/762fd3975c180db6fc88f577cf39549983fa373a/vilt/modules/objectives.py#L301
		# create soft encoding vectors for labels based on the labels and scores
		encoding['answers'] = annotation['labels']
		encoding['scores'] = annotation['scores']

		return encoding
	
def collate_fn_albef(batch):
	images = [item['image'] for item in batch]
	questions = [item['question'] for item in batch]
	answers = [x for lst in [item["answers"] for item in batch] for x in lst]
	#label_indices = [torch.tensor(item['label_indices']) for item in batch]
	weights = torch.tensor([x for lst in [item["scores"] for item in batch] for x in lst])
	n_answers = torch.tensor([len(item["answers"]) for item in batch])
	# create new batch
	batch = {}
	batch['image'] = torch.stack(images)
	batch['text_input'] = questions
	batch['answer'] = answers
	batch['weight'] = weights
	batch['n_answers'] = n_answers
	#batch['label_indices'] = padded_label_indices

	return batch