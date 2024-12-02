import os
import argparse
from collections import defaultdict
import random

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Function to parse the new formatted ratings data (user followed by a list of movie_ids)
def parse_ratings(file_path, max_len=200):
    user_data = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            user_id = parts[0]  
            movie_ids = parts[1:]  
            
            if len(movie_ids) > max_len:
                movie_ids = movie_ids[:max_len]
            
            user_data[user_id] = movie_ids
    return user_data

def split_data(user_data):
    train_data, valid_data, test_data = [], [], []
    counter = 0 
    for user_id, movie_ids in user_data.items():
        # print(user_id)
        # print(movie_ids)
        if len(movie_ids) >= 100 and counter < 6500:
            counter += 1
            test_data.append((user_id, ' '.join(movie_ids[:-1]), movie_ids[-1]))
            
            valid_data.append((user_id, ' '.join(movie_ids[:-2]), movie_ids[-2]))
            
            for i in range(1, len(movie_ids) - 2):
                train_data.append((user_id, ' '.join(movie_ids[:i]), movie_ids[i]))

    
    return train_data, valid_data, test_data

# Function to save the data in the desired format
def save_split_data(output_dir, split_name, data):
    check_path(output_dir)
    output_path = os.path.join(output_dir, f'{split_name}.inter')
    with open(output_path, 'w') as f:
        f.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for user_id, history, movie_id in data:
            f.write(f"{user_id}\t{history}\t{movie_id}\n")

def user_data_convert(user_data):
    train_data = []
    excluded_items = set()
    all_train_items = set()

    for user_id, movie_ids in user_data.items():
        train_data.append((user_id, ' '.join(movie_ids)))
        excluded_items.update(movie_ids[-2:])
        all_train_items.update(movie_ids[:-2])
        item_pool = list(all_train_items - excluded_items)

    return train_data, item_pool
    

def uniform_noise(file_path, corruption_param):
    user_data = parse_ratings(file_path)
    percentage = corruption_param / 100
    corrupt_train_data = defaultdict(list)

    converted_data, item_pool = user_data_convert(user_data)

    for user_id, item_list in converted_data:
        item_list_split = item_list.split()

        num_items_to_corrupt = int(len(item_list_split) * percentage)
        if num_items_to_corrupt > 0 and item_pool:
            indices_to_corrupt = random.sample(range(len(item_list_split)), num_items_to_corrupt)
            for idx in indices_to_corrupt:
                item_list_split[idx] = random.choice(item_pool)
        
        corrupt_train_data[user_id] = item_list_split

    return corrupt_train_data

#maybe change it to idx K
def recent_kth_noise(file_path, K):
    K = K + 2 #offset the train and validation
    user_data = parse_ratings(file_path)
    
    corrupt_train_data = defaultdict(list)

    converted_data, item_pool = user_data_convert(user_data)

    for user_id, item_list in converted_data:
        item_list_split = item_list.split()

        if item_pool:
            item_list_split[-K] = random.choice(item_pool)
        
        corrupt_train_data[user_id] = item_list_split

    return corrupt_train_data

def k_swap(file_path, K):
    user_data = parse_ratings(file_path)
    
    corrupt_train_data = defaultdict(list)

    converted_data, _ = user_data_convert(user_data)

    for user_id, item_list in converted_data:
        item_list_split = item_list.split()
        train_length = len(item_list_split) - 2

        for i in range(K):
            idx_1 = random.choice(range(train_length))
            idx_2 = random.choice(range(train_length))

            item_list_split[idx_1], item_list_split[idx_2] = item_list_split[idx_2], item_list_split[idx_1]
        
        corrupt_train_data[user_id] = item_list_split

    return corrupt_train_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratings_file', type=str, default='formatted_inter_ratings.inter', help='Path to formatted ratings data')
    parser.add_argument('--output_dir', type=str, default='output/', help='Directory to save split data')
    parser.add_argument('--max_len', type=int, default=200, help='Maximum number of ratings per user')
    parser.add_argument('--corrupt_type', type=str, default='none', help='type of corruptions on training dataset')
    parser.add_argument('--corrupt_K', type=int, default=2, help='corrupt the most recent Kth item')

    args = parser.parse_args()

    corruption = args.corrupt_type
    if  corruption == 'none':
        user_data = parse_ratings(args.ratings_file, args.max_len)
        train_data, valid_data, test_data = split_data(user_data)
        file_name = f'{args.corrupt_K}_{corruption}'
        save_split_data(args.output_dir, 'H&M_valid', valid_data)
        save_split_data(args.output_dir, 'H&M_test', test_data)
        save_split_data(args.output_dir, 'H&M_train', train_data)


        print("Data has been split and saved successfully!")
    elif corruption == 'uniform':
        corrupt_data = uniform_noise(args.ratings_file, args.corrupt_K)
        train_data, _, _ = split_data(corrupt_data)
        file_name = f'{args.corrupt_K}_uniform_train'
        save_split_data(args.output_dir, file_name, train_data)
        print("Uniform noises has been applied and saved successfully!")
    
    elif corruption == 'recentKth':
        corrupt_data = recent_kth_noise(args.ratings_file, args.corrupt_K)
        train_data, _, _ = split_data(corrupt_data)
        file_name = f'{args.corrupt_K}_recentKth_train'
        save_split_data(args.output_dir, file_name, train_data)
        print("Most recent K noises has been applied and saved successfully!")

    elif corruption == 'Kswap':
        corrupt_data = k_swap(args.ratings_file, args.corrupt_K)
        train_data, _, _ = split_data(corrupt_data)
        file_name = f'{args.corrupt_K}_Kswap_train'
        save_split_data(args.output_dir, file_name, train_data)
        print("K swap noises has been applied and saved successfully!")
