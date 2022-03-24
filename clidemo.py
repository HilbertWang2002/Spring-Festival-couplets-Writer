import torch
from module import Tokenizer, init_model_by_key
import argparse
import goto
from dominate.tags import label
from goto import with_goto
import sys
sys.setrecursionlimit(3000)
global isGo
isGo = True

def encode_decode(tokenizer, question, model, lens, device):
    global isGo
    if isGo ==False:
        return
    input_ids = torch.tensor(tokenizer.encode(question)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_ids).squeeze(0)
    pred = logits.argmax(dim=-1).tolist()
    pred = tokenizer.decode(pred)
    same = []
    flag=1
    whole_sentence = question+pred
    for i in range(lens):
        for j in range(i+1, lens):
            if question[i] == question[j]:
                same.append([i, j])
    # print(same)
    # print(pred)
    # print('intersection_bug', set(pred).intersection(question))
    if same == [] or set(pred).intersection(question)=={'，'}:
        # print(1)

        for i in range(lens):
            for j in range(i+1, lens):
                # if pred[i] == pred[j]:
                #     print('pred_bug')
                #
                # if (pred[i] == pred[j] and pred[j] != '，'):
                #     print(',error')
                # if((set(pred).intersection(question) != set() ) and (set(pred).intersection(question) != {'，'})):
                #     print(',_error')

                if(pred[i] == pred[j] and pred[j] != '，') or ((set(pred).intersection(question) != set() ) and (set(pred).intersection(question) != {'，'})):
                    # print('pred_different_bug')
                    encode_decode(tokenizer, question, model,lens,device)
                    flag =0
                    # goto .begin1
    else:
        # print(2)

            # print(same[i][0],same[i][1])
            for j in range(lens):
                for k in range(j+1,lens):
                    if( [j,k]  in same):
                        # if (j == same[i][0] and k == same[i][1]):
                        if(pred[j]!= pred[k] or pred[j] in question):
                            # print('pred_same_bug')
                            encode_decode(tokenizer, question, model, lens, device)
                            flag=0
                    else:
                        # if (pred[j] == pred[k] and pred[j] != '，'):
                        #     # print(',error')
                        if(pred[j] == pred[k] and pred[j] != '，') or ((set(pred).intersection(question) != set() ) and (set(pred).intersection(question) != {'，'})):
                            # print('pred_different_bug',pred[j])
                            encode_decode(tokenizer, question, model, lens, device)
                            flag=0

    if flag==1:
        print(f"下联：{pred}")

        isGo=False
        return

                # goto .begin1

# @with_goto
def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", default = r'output\Transformer_200.bin', type=str)
    # default = r'C:\Users\T JACK\Documents\GitHub\CoupletAI',
    parser.add_argument("-s", "--stop_flag", default='q', type=str)
    parser.add_argument("-c", "--cuda", action='store_true')
    args = parser.parse_args()
    print("loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    model_info = torch.load(args.path)
    tokenizer = model_info['tokenzier']
    model = init_model_by_key(model_info['args'], tokenizer)
    model.load_state_dict(model_info['model'])
    while True:
        global isGo
        isGo = True
        question = input("上联：")
        #new
        lens = len(question)
        # print(lens)
        for i in question:
            if i ==['，']:
                i=['，']
        if question == args.stop_flag.lower():
            print("Thank you!")
            break
        encode_decode(tokenizer,question,model,lens,device)
        # label .begin1
        # input_ids = torch.tensor(tokenizer.encode(question)).unsqueeze(0).to(device)
        # with torch.no_grad():
        #     logits = model(input_ids).squeeze(0)
        # pred = logits.argmax(dim=-1).tolist()
        # pred = tokenizer.decode(pred)
        # #new
        # same=[]
        # for i in range(lens):
        #     for j in range(i,lens):
        #         if question[i]==question[j]:
        #             same.append([i,j])
        # if same==None:
        #     for i in range(lens):
        #         for j in range(i,lens):
        #             if question[i]==question[j]:
        #                 encode_decode(tokenizer, question, model)
        #                 # goto .begin1
        # else:
        #     for i in range(len(same)):
        #         if pred[same[i][0]]!=pred[same[i][1]]:
        #             encode_decode(tokenizer, question, model)
        #             # goto .begin1



if __name__ == "__main__":
    run()
