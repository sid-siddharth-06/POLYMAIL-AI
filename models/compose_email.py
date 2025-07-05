from transformers import pipeline

# Load stronger transformer model
generator = pipeline("text2text-generation", model="google/flan-t5-large")

def compose_email_with_transformer(scenario: str) -> str:
    prompt = (
        f"Compose a formal email for the following request:\n\n"
        f"{scenario.strip()}\n\n"
        f"Structure it with a subject, greeting, body, and closing signature."
    )

    result = generator(
        prompt,
        max_new_tokens=256,
        temperature=0.7,
        repetition_penalty=1.5,
        do_sample=True,
        top_k=50
    )

    return result[0]['generated_text'].strip()

# Test it
if __name__ == "__main__":
    test = "I need to apply for two days leave next week due to personal reasons."
    email = compose_email_with_transformer(test)
    print("📧 Generated Email:\n", email)
