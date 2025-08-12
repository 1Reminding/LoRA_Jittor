import re
import matplotlib.pyplot as plt
import numpy as np

# Log file path
log_file = '/root/LoRA/NLG_jittor/trained_models/e2e/log1.txt'

# For storing extracted data
steps = []
train_losses = []
avg_losses = []
train_ppls = []
valid_losses = []
valid_ppls = []
lr_values = []

# Regular expression patterns
train_pattern = r'\| epoch\s+\d+\s+step\s+(\d+)\s+\|.*\| lr ([\d\.e-]+) \|.*\| loss\s+([\.\d]+)\s+\| avg loss\s+([\.\d]+)\s+\| ppl ([\d\.]+)'
eval_pattern = r'\| Eval\s+\d+\s+at step\s+(\d+).*\| valid loss\s+([\.\d]+)\s+\| valid ppl\s+([\.\d]+)\s+\| best ppl\s+([\.\d]+)'

# Read log file and extract data
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

# Create charts
plt.figure(figsize=(15, 20))

# 1. Training and validation loss
plt.subplot(4, 1, 1)
plt.plot(steps, train_losses, 'b-', label='Training Loss (Single Step)')
plt.plot(steps, avg_losses, 'g-', label='Average Training Loss')
# Add validation loss points
valid_steps, valid_loss_values = zip(*valid_losses) if valid_losses else ([], [])
plt.plot(valid_steps, valid_loss_values, 'ro-', label='Validation Loss')
plt.xlabel('Training Steps')
plt.ylabel('Loss Value')
plt.title('Loss Changes During Training')
plt.legend()
plt.grid(True)

# 2. Perplexity (PPL) chart
plt.subplot(4, 1, 2)
plt.plot(steps, train_ppls, 'b-', label='Training PPL')
# Add validation PPL points
valid_steps, valid_ppl_values = zip(*valid_ppls) if valid_ppls else ([], [])
plt.plot(valid_steps, valid_ppl_values, 'ro-', label='Validation PPL')
plt.xlabel('Training Steps')
plt.ylabel('Perplexity (PPL)')
plt.title('Perplexity Changes During Training')
plt.legend()
plt.grid(True)
plt.yscale('log')  # Use log scale to better display PPL changes

# 3. Learning rate changes
plt.subplot(4, 1, 3)
plt.plot(steps, lr_values, 'r-')
plt.xlabel('Training Steps')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Adjustment Curve')
plt.grid(True)

# 4. Smoothed training loss
window_size = 5
smoothed_losses = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
smoothed_steps = steps[window_size-1:]

plt.subplot(4, 1, 4)
plt.plot(smoothed_steps, smoothed_losses, 'g-', label='Smoothed Training Loss')
plt.xlabel('Training Steps')
plt.ylabel('Loss Value')
plt.title('Smoothed Training Loss Curve (Window Size={})'.format(window_size))
plt.legend()
plt.grid(True)

# Adjust layout and save
plt.tight_layout()
plt.savefig('/root/LoRA/NLG_jittor/trained_models/GPT2_M/e2e/training_curves.png', dpi=300)
plt.savefig('/root/LoRA/NLG_jittor/trained_models/GPT2_M/e2e/training_curves.pdf')
plt.show()

print(f"Charts saved to '/root/LoRA/NLG_jittor/trained_models/GPT2_M/e2e/training_curves.png' and '.pdf'")