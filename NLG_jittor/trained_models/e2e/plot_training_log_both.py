import re
import matplotlib.pyplot as plt
import numpy as np

# Log file paths
torch_log_file = '/root/LoRA/NLG_pytorch/trained_models/GPT2_M/e2e/log2.txt'
jittor_log_file = '/root/LoRA/NLG_jittor/trained_models/GPT2_M/e2e/log1.txt'

# For storing extracted data - PyTorch
torch_steps = []
torch_train_losses = []
torch_avg_losses = []
torch_train_ppls = []
torch_valid_losses = []
torch_valid_ppls = []
torch_lr_values = []

# For storing extracted data - Jittor
jittor_steps = []
jittor_train_losses = []
jittor_avg_losses = []
jittor_train_ppls = []
jittor_valid_losses = []
jittor_valid_ppls = []
jittor_lr_values = []

# Regular expression patterns
train_pattern = r'\| epoch\s+\d+\s+step\s+(\d+)\s+\|.*\| lr ([\d\.e-]+) \|.*\| loss\s+([\.\.\d]+)\s+\| avg loss\s+([\.\.\d]+)\s+\| ppl ([\d\.]+)'
eval_pattern = r'\| Eval\s+\d+\s+at step\s+(\d+).*\| valid loss\s+([\.\.\d]+)\s+\| valid ppl\s+([\.\.\d]+)\s+\| best ppl\s+([\.\.\d]+)'

# Function to extract data from log file
def extract_data_from_log(log_file, steps, train_losses, avg_losses, train_ppls, valid_losses, valid_ppls, lr_values):
    with open(log_file, 'r') as f:
        for line in f:
            # Extract training data
            train_match = re.search(train_pattern, line)
            if train_match:
                step = int(train_match.group(1))
                lr = float(train_match.group(2))
                loss = float(train_match.group(3))
                avg_loss = float(train_match.group(4))
                ppl = float(train_match.group(5))
                
                steps.append(step)
                train_losses.append(loss)
                avg_losses.append(avg_loss)
                train_ppls.append(ppl)
                lr_values.append(lr)
            
            # Extract validation data
            eval_match = re.search(eval_pattern, line)
            if eval_match:
                step = int(eval_match.group(1))
                valid_loss = float(eval_match.group(2))
                valid_ppl = float(eval_match.group(3))
                
                valid_losses.append((step, valid_loss))
                valid_ppls.append((step, valid_ppl))

# Extract data from both log files
extract_data_from_log(torch_log_file, torch_steps, torch_train_losses, torch_avg_losses, 
                     torch_train_ppls, torch_valid_losses, torch_valid_ppls, torch_lr_values)
extract_data_from_log(jittor_log_file, jittor_steps, jittor_train_losses, jittor_avg_losses, 
                     jittor_train_ppls, jittor_valid_losses, jittor_valid_ppls, jittor_lr_values)

# Create charts
plt.figure(figsize=(15, 20))

# 1. Training loss comparison
plt.subplot(4, 1, 1)
plt.plot(torch_steps, torch_avg_losses, 'b-', label='PyTorch Avg Training Loss')
plt.plot(jittor_steps, jittor_avg_losses, 'r-', label='Jittor Avg Training Loss')
# Add validation loss points
torch_valid_steps, torch_valid_loss_values = zip(*torch_valid_losses) if torch_valid_losses else ([], [])
jittor_valid_steps, jittor_valid_loss_values = zip(*jittor_valid_losses) if jittor_valid_losses else ([], [])
plt.plot(torch_valid_steps, torch_valid_loss_values, 'bo-', label='PyTorch Validation Loss')
plt.plot(jittor_valid_steps, jittor_valid_loss_values, 'ro-', label='Jittor Validation Loss')
plt.xlabel('Training Steps')
plt.ylabel('Loss Value')
plt.title('Loss Comparison: PyTorch vs Jittor')
plt.legend()
plt.grid(True)

# 2. Perplexity (PPL) comparison
plt.subplot(4, 1, 2)
plt.plot(torch_steps, torch_train_ppls, 'b-', label='PyTorch Training PPL')
plt.plot(jittor_steps, jittor_train_ppls, 'r-', label='Jittor Training PPL')
# Add validation PPL points
torch_valid_steps, torch_valid_ppl_values = zip(*torch_valid_ppls) if torch_valid_ppls else ([], [])
jittor_valid_steps, jittor_valid_ppl_values = zip(*jittor_valid_ppls) if jittor_valid_ppls else ([], [])
plt.plot(torch_valid_steps, torch_valid_ppl_values, 'bo-', label='PyTorch Validation PPL')
plt.plot(jittor_valid_steps, jittor_valid_ppl_values, 'ro-', label='Jittor Validation PPL')
plt.xlabel('Training Steps')
plt.ylabel('Perplexity (PPL)')
plt.title('Perplexity Comparison: PyTorch vs Jittor')
plt.legend()
plt.grid(True)
plt.yscale('log')  # Use log scale to better display PPL changes

# 3. Learning rate comparison
plt.subplot(4, 1, 3)
plt.plot(torch_steps, torch_lr_values, 'b-', label='PyTorch Learning Rate')
plt.plot(jittor_steps, jittor_lr_values, 'r-', label='Jittor Learning Rate')
plt.xlabel('Training Steps')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Comparison: PyTorch vs Jittor')
plt.legend()
plt.grid(True)

# 4. Smoothed training loss comparison
window_size = 5
torch_smoothed_losses = np.convolve(torch_train_losses, np.ones(window_size)/window_size, mode='valid')
torch_smoothed_steps = torch_steps[window_size-1:]
jittor_smoothed_losses = np.convolve(jittor_train_losses, np.ones(window_size)/window_size, mode='valid')
jittor_smoothed_steps = jittor_steps[window_size-1:]

plt.subplot(4, 1, 4)
plt.plot(torch_smoothed_steps, torch_smoothed_losses, 'b-', label='PyTorch Smoothed Training Loss')
plt.plot(jittor_smoothed_steps, jittor_smoothed_losses, 'r-', label='Jittor Smoothed Training Loss')
plt.xlabel('Training Steps')
plt.ylabel('Loss Value')
plt.title('Smoothed Training Loss Comparison (Window Size={})'.format(window_size))
plt.legend()
plt.grid(True)

# Adjust layout and save
plt.tight_layout()
plt.savefig('/root/LoRA/NLG_jittor/trained_models/GPT2_M/e2e/training_comparison.png', dpi=300)
plt.savefig('/root/LoRA/NLG_jittor/trained_models/GPT2_M/e2e/training_comparison.pdf')
plt.show()

print(f"Comparison charts saved to '/root/LoRA/NLG_jittor/trained_models/GPT2_M/e2e/training_comparison.png' and '.pdf'")