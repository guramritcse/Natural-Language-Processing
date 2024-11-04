from metric import load_model

import json

def read_json(file_path):
    entries = []
    # Open the JSON file and read the entries
    with open(file_path, "r") as json_file:
        for line in json_file:
            entry = json.loads(line.strip())
            entries.append(entry)
    return entries

def main():
    # Specify the path to the JSON file
    json_file_path = "results/chat_results.json"

    model = load_model("results/model.pkl")

    # Read entries from the JSON file
    entries = read_json(json_file_path)

    # Process the entries
    correct_tokens_pred = 0
    total_tokens = 0
    correct_sentences_pred = 0
    total_sentences = 0
    correct_tokens_chat = 0
    correct_sentences_chat = 0
    for i, entry in enumerate(entries):
        all_correct_pred = True
        all_correct_chat = True
        pred = model.predict([entry['sentence']])[0]
        for predicted_tag, actual_tag, chat_tag in zip(pred, entry['label'], entry['chatlabel']):
            total_tokens += 1
            if predicted_tag == actual_tag:
                correct_tokens_pred += 1
            else:
                all_correct_pred = False
            
            if chat_tag == actual_tag:
                correct_tokens_chat+=1
            else:
                all_correct_chat = False

        if all_correct_pred:
            correct_sentences_pred += 1
        if all_correct_chat:
            correct_sentences_chat+=1
        total_sentences += 1
    token_acc_pred = correct_tokens_pred / total_tokens
    sentence_acc_pred = correct_sentences_pred / total_sentences
    token_acc_chat = correct_tokens_chat / total_tokens
    sentence_acc_chat = correct_sentences_chat / total_sentences
    test_sentences = len(entries)
    print(f"For Our Model - Token accuracy: {token_acc_pred}, Sentence accuracy: {sentence_acc_pred}, Total tokens: {total_tokens}, Total sentences: {total_sentences}, Correct tokens: {correct_tokens_pred}, Correct sentences: {correct_sentences_pred}, Test sentences: {test_sentences}")
    print(f"For ChatGPT - Token accuracy: {token_acc_chat}, Sentence accuracy: {sentence_acc_chat}, Total tokens: {total_tokens}, Total sentences: {total_sentences}, Correct tokens: {correct_tokens_chat}, Correct sentences: {correct_sentences_chat}, Test sentences: {test_sentences}")
    



if __name__ == "__main__":
    main()
