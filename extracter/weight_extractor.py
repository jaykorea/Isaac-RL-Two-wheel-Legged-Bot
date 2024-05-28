import torch

# Helper function to categorize weights
def get_category_description(layer_name):
    if 'weight' in layer_name:
        return 'Weights'
    elif 'bias' in layer_name:
        return 'Biases'
    elif 'running_mean' in layer_name:
        return 'Running Mean'
    elif 'running_var' in layer_name:
        return 'Running Variance'
    elif 'step' in layer_name:
        return 'Optimizer Step'
    elif 'exp_avg' in layer_name:
        return 'Exponential Moving Average'
    elif 'exp_avg_sq' in layer_name:
        return 'Exponential Moving Average of Squared Values'
    else:
        return 'Unknown Category'

# Save tensor to file
def save_tensor_to_file(tensor, f):
    tensor_np = tensor.cpu().numpy()
    tensor_shape = tensor_np.shape
    f.write(f"Weights shape: {tensor_shape}\n")
    for value in tensor_np.flatten():
        f.write(f"{value:.6e} ")
    f.write("\n")

# Save state dictionary to file with categorization
def save_state_dict_to_file(state_dict, output_file):
    with open(output_file, 'w') as f:
        for key, value in state_dict.items():
            category = get_category_description(key)
            f.write(f"Key: {key} ({category})\n")
            if isinstance(value, torch.Tensor):
                save_tensor_to_file(value, f)
            elif isinstance(value, dict):
                f.write(f"{key} (Nested Dictionary):\n")
                save_optimizer_state_to_file(value, f, indent=2)
            else:
                f.write(f"{value}\n")
            f.write('-' * 60 + '\n')

# Save additional optimizer state dictionary if present
def save_optimizer_state_to_file(state_dict, f, indent=0):
    for key, value in state_dict.items():
        f.write(' ' * indent + f"{key}: ")
        if isinstance(value, torch.Tensor):
            f.write("\n")
            save_tensor_to_file(value, f)
        elif isinstance(value, dict):
            f.write("\n")
            save_optimizer_state_to_file(value, f, indent + 2)
        else:
            f.write(f"{value}\n")

# Example usage
pth_file = 'Flamingo.pth'
output_file = 'model_weights.txt'

# Load the state dictionary from the .pth file
state_dict = torch.load(pth_file)
if 'state_dict' in state_dict:
    state_dict = state_dict['state_dict']

save_state_dict_to_file(state_dict, output_file)

