import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

torch.manual_seed(42)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data)*0.9)
train_data = data[:n]
val_data = data[n:]


def get_batch(train):
    data = train_data if train else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    xb = torch.stack([data[i:i+block_size] for i in ix])
    yb = torch.stack([data[i+1:i+block_size+1] for i in ix])
    xb, yb = xb.to(device), yb.to(device)
    return xb, yb

@torch.no_grad
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # input (B,T)
        # targets (B,T)

        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # input (B,T)

        for _ in range(max_new_tokens):
            logits, loss = self(idx) # (B,T,C)
            logits = logits[:, -1, :] # (B,C)
            probs = F.softmax(logits, dim=1) # (B,C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.concat((idx, idx_next), dim=1) # (B,T+1)

        return idx
        

model = BigramLanguageModel(vocab_size=vocab_size)
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'Step {iter}: train loss = {losses["train"]}, val loss = {losses["val"]}')

    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

'''
Step 0: train loss = 4.762716770172119, val loss = 4.750523090362549
Step 300: train loss = 2.8414525985717773, val loss = 2.831892728805542
Step 600: train loss = 2.5515248775482178, val loss = 2.55210018157959
Step 900: train loss = 2.4970433712005615, val loss = 2.506361722946167
Step 1200: train loss = 2.490762233734131, val loss = 2.4806456565856934
Step 1500: train loss = 2.4716250896453857, val loss = 2.472615957260132
Step 1800: train loss = 2.45572566986084, val loss = 2.472358226776123
Step 2100: train loss = 2.4724764823913574, val loss = 2.4694225788116455
Step 2400: train loss = 2.463688611984253, val loss = 2.4528136253356934
Step 2700: train loss = 2.4643714427948, val loss = 2.458897113800049
'''