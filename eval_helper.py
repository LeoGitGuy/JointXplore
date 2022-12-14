import torch
from tqdm import tqdm

def compute_metrics(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return labels[torch.arange(labels.size(0)), preds].sum()/labels.shape[0]

def compute_metrics_albef(preds, batch):
    acc = 0
    al = 0
    answer_scores = list(zip(batch["answer"], batch["weight"]))
    for idx, pred in enumerate(preds):
        ah = al + batch["n_answers"][idx].item()
        match = [t[1] for t in answer_scores[al:ah] if t[0] == pred]
        acc += next(iter(match), torch.tensor(0)).item()
        al = ah
    acc = acc/len(preds)
    return acc

def compute_topk_acc(output, target_labels, topk=(1,)):
    maxk = max(topk)
    batch_size = target_labels.size(0)
    _, y_pred = output.topk(k=maxk, dim=1)
    y_pred = y_pred.t()  
    topk_acc = []
    for k in range(maxk):
        y_pred_reshaped = y_pred[k][:,None].expand(-1, target_labels.shape[1])
        correct_topk_in_batch = (target_labels==y_pred_reshaped).sum()
        if k == 0:
            topk_acc.append(correct_topk_in_batch)
        else:
            topk_acc.append(correct_topk_in_batch + topk_acc[-1])
        chosen_topk_accs = [topk_acc[j-1] for j in topk]
    chosen_topk_accs = torch.tensor(chosen_topk_accs)
    return chosen_topk_accs

def eval_model(model, dataloader, device, model_type, answer_list=[], topk = (1,3,5)):
    model.eval()
    total_loss = 0
    total_acc = 0
    topk_accs = torch.tensor(len(topk)*[0])
    for batch in tqdm(dataloader):
        # get the inputs; 
        #
        #label_indices = batch.pop("label_indices")
        with torch.no_grad():
            if model_type == "vilt":
                batch = {k:v.to(device) for k,v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                total_acc += compute_metrics(outputs.logits, batch["labels"]).item()
            elif model_type == "albef":
                preds = model.predict_answers(samples = {"image": batch["image"], "text_input": batch["text_input"]},
                                      answer_list = answer_list)
                total_acc += compute_metrics_albef(preds, batch)
            #topk_accs = torch.add(topk_accs, compute_topk_acc(outputs.logits, label_indices, topk))
    total_acc = total_acc / len(dataloader)
    total_loss = total_loss / len(dataloader)
    #topk_accs = topk_accs / len(dataset)
    return total_loss, total_acc
