from enum import Enum
from torch import nn
import torch
import pickle
import copy

Modes = Enum('Modes', ['REGION', 'KSECTION', 'BOUNDARY', 'STRONG'])
Act_fct = Enum('Act_fct', ['GELU', 'TANH'])
hooks = []
activations = {}

# Creating a hook
def create_hook(name, out, modes, model_type, activation, k=10):
    def hook(module, in_tensor, out_tensor):
        out_tensor = copy.deepcopy(out_tensor.detach())
        # Only use this mode with training data
        if (model_type == "albef" and out_tensor.shape == torch.Size([4, 1, 768])) or model_type=="vilt":
            if Modes.REGION in modes:
                # initialize boundary tensors
                if not "low" in out:
                    out["low"] = {}
                    out["high"] = {}
                if not name in out["low"]:
                    out["low"][name] = torch.min(out_tensor, dim=0).values
                    out["high"][name] = torch.max(out_tensor, dim=0).values
                # use min, max to get boundary values in each run after initialization
                else:
                    out["low"][name] =torch.min(out["low"][name], torch.min(out_tensor, dim=0).values)
                    out["high"][name] = torch.max(out["high"][name], torch.max(out_tensor, dim=0).values)

            # Computes K-Section Coverage
            if Modes.KSECTION in modes:
                mode = Modes.KSECTION
                try:
                    assert "low" in out
                    assert name in out["low"]
                except:
                    raise Exception('Calculate boundary dicts low and max first')
                step_size = (out["high"][name]-out["low"][name])/k
                for step in range(k):
                    # initialize k-section tensors
                    if not mode.name in out:
                        out[mode.name] = {}
                    if not name in out[mode.name]:
                        out[mode.name][name] = {}
                    if not step in out[mode.name][name]:
                        out[mode.name][name][step] = torch.zeros(out_tensor.shape[1:])
                    # if neuron is in range between the k-1th and kth section, add +1 to the kth section tensor
                    out[mode.name][name][step] = out[mode.name][name][step].add(
                            torch.sum(
                            torch.logical_and(
                                    torch.ge(out_tensor, out["low"][name]+(step*step_size)),
                                    torch.lt(out_tensor, out["low"][name]+((step+1)*step_size)))
                            .long(), axis=0))
            
            # Computes Boundary Coverage
            if Modes.BOUNDARY in modes:
                mode = Modes.BOUNDARY
                try:
                    assert "low" in out
                    assert name in out["low"]
                except:
                    raise Exception('Calculate boundary dicts low and max first')
                # initialize strong/weak tensors
                if not mode.name in out:
                    out[mode.name] = {}
                if not name in out[mode.name]:
                    out[mode.name][name] = {}
                if not "strong" in out[mode.name][name]:
                    out[mode.name][name]["strong"] = torch.zeros(out_tensor.shape[1:])        
                    out[mode.name][name]["weak"] = torch.zeros(out_tensor.shape[1:])
                # if neuron is above max or below low boundaries, add +1 to the respective strong/weak tensor
                out[mode.name][name]["strong"] = out[mode.name][name]["strong"].add(
                        torch.sum(
                                torch.gt(out_tensor, out["high"][name]), axis=0))
                out[mode.name][name]["weak"] = out[mode.name][name]["weak"].add(
                        torch.sum(
                                torch.lt(out_tensor, out["low"][name]), axis=0))
    return hook

def read_activations(activations_file):
    global activations
    with open(activations_file, "rb") as fp:
            activations = pickle.load(fp)
    return activations

def get_activations():
    return activations

# Loop through all layers of the model and choose the ones with an activaton function to register a hook
def get_all_activation_layers(net, modes, model_type, k=10):
    if model_type == "vilt":
            for name, layer in net._modules.items():
                    if name == "vilt":
                            n3, l3 = list(list(layer._modules.items())[3][1]._modules.items())[1]
                            l3.register_forward_hook(create_hook(n3, activations, modes, model_type, Act_fct.TANH, k))
                    else:
                            n4, l4 = list(layer._modules.items())[2]
                            l4.register_forward_hook(create_hook(n4, activations, modes, model_type, Act_fct.GELU, k))
    elif model_type == "albef":
            mod = net.text_decoder.cls.predictions.transform.transform_act_fn
            hooks.append(
                    mod.register_forward_hook(
                    create_hook("text_decoder", activations, modes, model_type, Act_fct.GELU, k)))

def compute_ksection_coverage(activations):
        if not "COVERAGE" in activations:
                activations["COVERAGE"] = {}
        if not "KSECTION" in activations["COVERAGE"]:
                activations["COVERAGE"]["KSECTION"] = {}
                
        for key in activations["KSECTION"].keys():
                activations["COVERAGE"]["KSECTION"][key] = 0
                for i in list(activations["KSECTION"][key].keys()):
                        activations["COVERAGE"]["KSECTION"][key] += (torch.sum(activations['KSECTION'][key][i] > 0)/len(activations['KSECTION'][key][i]))
                activations["COVERAGE"]["KSECTION"][key] = (activations["COVERAGE"]["KSECTION"][key]/len(list(activations["KSECTION"][key].keys()))).item()
                print(f"KSECTION COVERAGE, '{key}'-Activation: {activations['COVERAGE']['KSECTION'][key]}")

        return activations

def compute_boundary_strong_coverage(activations):
        if not "COVERAGE" in activations:
                activations["COVERAGE"] = {}
        if not "BOUNDARY" in activations["COVERAGE"]:
                activations["COVERAGE"]["BOUNDARY"] = {}
        if not "STRONG" in activations["COVERAGE"]:
                activations["COVERAGE"]["STRONG"] = {}
                
        for key in activations["BOUNDARY"].keys():
                strong = torch.sum(activations['BOUNDARY'][key]["strong"] > 0)/len(activations['BOUNDARY'][key]['strong'])
                weak =  torch.sum(activations['BOUNDARY'][key]["weak"] > 0)/len(activations['BOUNDARY'][key]['weak'])
                activations["COVERAGE"]["BOUNDARY"][key] = ((strong+weak)/2).item()
                activations["COVERAGE"]["STRONG"][key] = strong.item()
                print(f"BOUNDARY COVERAGE, '{key}'-Activation: {activations['COVERAGE']['BOUNDARY'][key]}")
                print(f"STRONG COVERAGE, '{key}'-Activation: {activations['COVERAGE']['STRONG'][key]}")
        return activations
