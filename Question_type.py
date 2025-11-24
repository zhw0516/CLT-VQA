import random
import json
import numpy
seed = 66666

random.seed(seed)
print('random seed', seed)

def random_dic(dicts):
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    new_dict = {}
    for key in dict_key_ls:
        new_dict[key] = dicts.get(key)
    return new_dict

All_task = ['q_recognition', 'q_count', 'q_color', 'q_subcategory','q_action', 'q_commonsense', 'q_type', 'q_location','q_causal']

Category_splits = {'G1': [58, 48, 55, 36, 64, 1, 70, 73, 42, 15, 6, 18, 49, 59, 31, 2],\
                   'G2': [19, 77, 22, 9, 24, 53, 12, 13, 78, 50, 47, 41, 32, 28, 54, 23],\
                   'G3': [60, 8, 34, 25, 67, 4, 14, 68, 3, 79, 0, 5, 65, 20, 71, 39], \
                   'G4': [35, 29, 66, 40, 43, 26, 72, 10, 38, 61, 76, 44, 75, 69, 16, 57], \
                   'G5': [45, 33, 63, 56, 21, 11, 62, 74, 17, 52, 46, 30, 27, 51, 37, 7]}




with open('datasets/QuesId_task_map.json') as fp:
    QuesId_task_map = json.load(fp)

with open('datasets/ImgId_cate_map.json') as fp:
    ImgId_cate_map = json.load(fp)

print("Success to load the QuesId_task_map and QuesId_task_map")


def save_results_matrix(results, path):
    dict_json = json.dumps(results)
    with open(path, 'w') as file:
        file.write(dict_json)
    print('The result matrix has been wrote into ', path)
def save_results(results, path):
    dict_json = json.dumps(results)
    with open(path, 'w') as file:
        file.write(dict_json)
    print('The result has been wrote into ', path)
def show_results_matrix(results, start=0):
    matrix = numpy.zeros([len(results), len(results)], dtype=float)
    key_list = []
    for key in results:
        print(key, end='\t')
        key_list.append(key)
    print('\n')
    for i in range(start,len(results)):
        avg = 0
        # print('T1   ', end='\t')
        for j in range(start,len(results)):
            if j < i+1:
                matrix[i][j] = results[key_list[i]][key_list[j]]
                avg += matrix[i][j]
            print(round(matrix[i][j], 2), end='\t')

        print("Avg:", round(avg / (len(results)-start),2))


def evaluate_metric(results, start=0):
    matrix = numpy.zeros([len(results), len(results)], dtype=float)-1
    key_list = []
    for key in results:
        key_list.append(key)
    for i in range(start, len(results)):
        avg = 0
        for j in range(start, len(results)):
            if j < i + 1:
                matrix[i][j] = results[key_list[i]][key_list[j]]

    Incre_avg_accuracy = []
    for t in range(start, len(results)):
        now_acc = matrix[t]
        all_acc = 0
        num = 0
        for acc in now_acc:
            if acc != -1:
                all_acc += acc
                num += 1
        avg_acc = all_acc / num
        Incre_avg_accuracy.append(avg_acc)


    Avg_accuracy = Incre_avg_accuracy[-1]

    Incre_avg_forget = [0]
    for t in range(1+start,len(results)):
        results_now = matrix[:t+1, :t+1]
        t_forget = []
        for idx in range(start, len(results_now)-1):
            task_list = results_now[:-1,idx]
            final = results_now[-1,idx]
            pre_max = max(task_list)
            if pre_max == -1:
                t_forget.append(0)
            else:
                t_forget.append(pre_max - final)
        Avg_forget = sum(t_forget)/len(t_forget)
        Incre_avg_forget.append(Avg_forget)


    Avg_forget = Incre_avg_forget[-1]


    output_dict = {'Incre_avg_acc': Incre_avg_accuracy,
                   'Avg_acc': Avg_accuracy,
                   'Incre_avg_forget': Incre_avg_forget,
                   'Avg_forget':Avg_forget
                   }
    return output_dict