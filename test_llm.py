import torch
import tiktoken
import torch.nn as nn
import json

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

# --- Input formatter (Uses stage name directly) ---
def format_input(stage_name):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\nParkinson's Stage: {stage_name}" # Use stage name here
    )
    return instruction_text

# --- Output parser ---
def parse_structured_output(raw_output):
    return {"answer": raw_output.strip()}

# --- Run inference ---
def run_inference_by_stage(stage_name, model, tokenizer, device, base_config):
    # Validate the input stage name against the known stages
    valid_stages = ["Normal", "Mild PD", "Moderate PD", "Severe PD", "Very Severe PD"]
    if stage_name not in valid_stages:
        return {"answer": f"Invalid Parkinson's Stage provided. Please enter one of: {', '.join(valid_stages)}."}

    print(f"Using Parkinson's Stage: {stage_name}")

    # 1. Format input text using the stage name
    input_text = format_input(stage_name)
    tokens = text_to_token_ids(input_text, tokenizer).to(device)
    print("🔤 Token IDs:", tokens)
    print("🧠 Token max ID:", int(tokens.max()), " | Vocab size:", model.tok_emb.num_embeddings)

    # 2. Generate text from the model
    token_ids = generate(
        model=model,
        idx=tokens,
        max_new_tokens=256, # Increased max_new_tokens for potentially longer responses
        context_size=base_config["context_length"],
        eos_id=50256 # Assuming 50256 is the EOS token ID for gpt2
    )

    generated_text = token_ids_to_text(token_ids, tokenizer)
    print("Generated text before parsing:\n", generated_text)

    # 3. Parse the generated output
    response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

    return parse_structured_output(response_text)

# --- Main execution block for testing ---
if __name__ == "__main__":
    # Load tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Model configuration (Ensure this matches the training config)
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0, # Set dropout to 0 for inference
        "qkv_bias": True
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12}
    }

    CHOOSE_MODEL = "gpt2-small (124M)"
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(BASE_CONFIG)

    # Load the fine-tuned model state dictionary
    try:
        # Note: Update the path to your fine-tuned model checkpoint if it's different
        checkpoint = torch.load("MODEL PATH", map_location=device)
        model.load_state_dict(checkpoint["Newmodel_state_dict"])
        model.to(device)
        model.eval() # Set model to evaluation mode
        print("✅ Fine-tuned model loaded successfully.")
    except FileNotFoundError:
        print("❌ Error: Fine-tuned model checkpoint not found. Please ensure 'Newfinal_parkinsons_stage_model.pth' exists.")
        exit()
    except KeyError:
         print("❌ Error: 'Newmodel_state_dict' not found in the checkpoint file. Ensure the correct key is used.")
         exit()


    # Interactive test loop
    while True:
        try:
            stage_input = input("Enter Parkinson's Stage (Normal, Mild PD, Moderate PD, Severe PD, Very Severe PD, or 'quit' to exit): ").strip()
            if stage_input.lower() == 'quit':
                break

            # Run inference with the provided stage name
            result = run_inference_by_stage(stage_input, model, tokenizer, device, BASE_CONFIG)
            print("\n🧠 Model Insight:\n", result["answer"])

        except Exception as e:
            print(f"❌ An error occurred during inference: {e}")