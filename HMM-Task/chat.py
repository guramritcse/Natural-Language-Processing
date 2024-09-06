import json
import os 

filename = 'datasets/0_wrong.json'
with open(filename, 'r') as f:
    wrong_statements = json.load(f)

chatgpt_outputs = {}

# if filename exists, then delete
if '0_wrong_chatgpt_outputs.json' in os.listdir("datasets"):
    os.remove("datasets/0_wrong_chatgpt_outputs.json")

tag_to_index = {'ADJ': 0, 'ADP': 1, 'ADV': 2, 'CONJ': 3, 'DET': 4, 'NOUN': 5, 'NUM': 6, 'PRON': 7, 'PRT': 8, 'VERB': 9, '.': 10, 'X': 11}
tagset = list(tag_to_index.keys())

start = True
for i in list(wrong_statements.keys())[:50]:
    print("Statement:", i)
    print(wrong_statements[str(i)]['words'])
    while(True):
        print("Enter the tags:")
        tags = input()
        eachtags = tags.split()
        if len(eachtags) != len(wrong_statements[str(i)]['words'].split()):
            print("Number of tags should be equal to number of words in the statement")
            print("Number of words in the statement:", len(wrong_statements[str(i)]['words'].split()))
            print("Number of tags entered:", len(eachtags))
            continue
        elif not all([tag in tagset for tag in eachtags]):
            print("Tags should be from the tagset")
            continue
        chatgpt_outputs[i] = {
            "words": wrong_statements[str(i)]['words'],
            "actual_tags": wrong_statements[str(i)]['actual_tags'],
            "chatgpt_tags": tags
        }
        break

    if start:
        start = False
        with open("datasets/0_wrong_chatgpt_outputs.json", "w") as f:
            json.dump(chatgpt_outputs, f, indent=4)
    else:
        with open("datasets/0_wrong_chatgpt_outputs.json", "r") as f:
            curr = json.load(f)
        curr.update(chatgpt_outputs)
        with open("datasets/0_wrong_chatgpt_outputs.json", "w") as f:
            json.dump(curr, f, indent=4)


