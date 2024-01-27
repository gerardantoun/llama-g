import fire
from typing import Optional
from llama import Llama, Dialog

def main(
    ckpt_dir: str, 
    tokenizer_path: str, 
    temperature: float = 0.6, 
    top_p: float = 0.9, 
    max_seq_len: int = 128, 
    max_gen_len: int = 128, 
    max_batch_size: int = 1,
    messages: Optional[list] = None
):
    print("Building generator...")
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    print("Generator built. Type 'quit' to exit.")

    if messages is None:
        messages = [{"role": "system", "content": "Start of conversation"}]

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        dialog = messages + [{"role": "user", "content": user_input}]
        messages = dialog 
        input(f"{dialog}\nPress Enter to continue...")
        result = generator.chat_completion(
            [dialog],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )[0]
        print(f"Assistant: {result['generation']['content']}")
        messages.append({"role": "assistant", "content": result['generation']['content']})
if __name__ == "__main__":
    fire.Fire(main)