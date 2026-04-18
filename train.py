import json, os, torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments, Trainer
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "/content/model"
DATA_FILE = "/content/data.json"
MAX_LEN = 128

os.environ["WANDB_DISABLED"] = "true"

CATEGORIES = [
    "billing","order_management","returns_refunds","account_access",
    "technical_support","general_inquiry","urgent_escalation","feedback",
]

PRIORITY_MAP = {
    "billing":"P2","order_management":"P3","returns_refunds":"P3",
    "account_access":"P2","technical_support":"P2",
    "general_inquiry":"P4","urgent_escalation":"P1","feedback":"P4",
}

SUGGESTED_ACTIONS = {
    "billing": "Escalate to billing team within 24 hours.",
    "order_management": "Check order system and provide tracking update.",
    "returns_refunds": "Initiate return and process refund within 5-7 days.",
    "account_access": "Verify identity and send password reset link.",
    "technical_support": "Assign to Tier-2 and schedule callback within 2 hours.",
    "general_inquiry": "Provide FAQ link or knowledge base article.",
    "urgent_escalation": "Escalate to senior manager immediately.",
    "feedback": "Log in CRM and send acknowledgement within 1 hour.",
}

SUMMARIES = {
    "billing": "Customer has a billing or payment issue.",
    "order_management": "Customer needs help with an order.",
    "returns_refunds": "Customer wants a return or refund.",
    "account_access": "Customer cannot access their account.",
    "technical_support": "Customer has a technical issue.",
    "general_inquiry": "Customer has a general question.",
    "urgent_escalation": "Critical issue needing immediate escalation.",
    "feedback": "Customer is providing feedback.",
}

PROMPT = """### Support Ticket:
{ticket}
### Instructions:
Classify and return ONLY valid JSON with keys: category, priority, summary, suggested_action.
Categories: {cats}

### Response:
"""

# -------------------- DATASET --------------------

def build_dataset():
    with open(DATA_FILE) as f:
        tickets = json.load(f)["tickets"]

    rows = []
    for t in tickets:
        cat = t["category"]

        prompt = PROMPT.format(
            ticket=t["text"],
            cats=", ".join(CATEGORIES)
        )

        resp = json.dumps({
            "category": cat,
            "priority": PRIORITY_MAP[cat],
            "summary": SUMMARIES[cat],
            "suggested_action": SUGGESTED_ACTIONS[cat],
        })

        rows.append({"text": prompt + resp})

    return rows


def tokenise(rows, tokenizer):
    ids_list, mask_list, lbl_list = [], [], []

    for row in rows:
        enc = tokenizer(
            row["text"],
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        ids = enc["input_ids"][0]
        mask = enc["attention_mask"][0]

        labels = ids.clone()
        labels[mask == 0] = -100

        ids_list.append(ids.tolist())
        mask_list.append(mask.tolist())
        lbl_list.append(labels.tolist())

    return Dataset.from_dict({
        "input_ids": ids_list,
        "attention_mask": mask_list,
        "labels": lbl_list
    })


# -------------------- MAIN --------------------

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading model in 4-bit...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model)

    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            inference_mode=False,
        )
    )

    model.print_trainable_parameters()

    print("Preparing data...")
    rows = build_dataset()
    split = int(len(rows) * 0.8)

    train_d = tokenise(rows[:split], tokenizer)
    val_d = tokenise(rows[split:], tokenizer)

    print(f"Train: {len(train_d)} | Val: {len(val_d)}")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=5,
        eval_strategy="epoch",   # ✅ fixed
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=[],
        optim="paged_adamw_32bit",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_d,
        eval_dataset=val_d,
    )

    print("Training started...")
    trainer.train()

    os.makedirs(f"{OUTPUT_DIR}/final", exist_ok=True)
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

    print("Model saved successfully!")


# -------------------- RUN --------------------

if __name__ == "__main__":
    main()
