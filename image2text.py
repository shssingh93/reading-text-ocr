from PIL import Image, ImageDraw, ImageFont
import sys
import re
import math
import heapq
import copy

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    print(im.size)
    print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

#Function to derive initial probabilities
def get_initial_prob(train_letters, test_letters, initial_prob_dict, train_txt_fname):

    train_file = open(train_txt_fname, 'r')   #Open bc.train file in read mode to calculate initial probabilities

    for line in train_file:

        # We have considered only evenly positioned words in the file since the bc.train also has corresponding POS tags for each
        # word in the file
        data = [line.split()[0::2]] 

        if data[0][0][0] in ('`', '&', '$', ';', '*', '!', '?', '\''):
            continue
        elif data[0][0][0] in initial_prob_dict:
            initial_prob_dict[data[0][0][0]] += 1
        else:
            initial_prob_dict[data[0][0][0]] = 1
    
    total_letter_count = sum(initial_prob_dict.values())

    for key in initial_prob_dict:
        initial_prob_dict[key] = math.log(initial_prob_dict[key])/total_letter_count


#Function to derive transition probabilities for each pair of consecutive letters in the file (transition pairs) in the training file
def get_transition_prob(train_letters, test_letters, transition_prob_dict, train_txt_fname):

    train_file = open(train_txt_fname, 'r')   #Open bc.train file in read mode to calculate transition probabilities

    for line in train_file:

        # We have considered only evenly positioned words in the file since the bc.train also has corresponding POS tags for each
        # word in the file
        data = list(" ".join([word for word in line.split()][0::2])) 

        # Calculating the count of each consecutive letter pair in the training file for deriving transition probability
        for cur_letter in range(1, len(data)):

            if data[cur_letter] in transition_prob_dict:
                if data[cur_letter - 1] in transition_prob_dict[data[cur_letter]]:
                    transition_prob_dict[data[cur_letter]][data[cur_letter - 1]] += 1
                else:
                    transition_prob_dict[data[cur_letter]][data[cur_letter - 1]] = 1
            else:
                transition_prob_dict[data[cur_letter]] = {data[cur_letter - 1] : 1}

    # Calculating the transition probability by dividing the above calculated count with total no of occurences for the current letter
    for cur_letter in transition_prob_dict:
        total_letter_count = sum(transition_prob_dict[cur_letter].values())
        for next_letter in transition_prob_dict[cur_letter]:
            transition_prob_dict[cur_letter][next_letter] = math.log(transition_prob_dict[cur_letter][next_letter]) / total_letter_count



# Function to derive emission probabilities on the basis of matching pixels between observed characters in the test image and actual 
# characters in the train image
def get_emission_prob(train_letters, test_letters, emission_prob_dict):

   for test_letter in range(len(test_letters)):

    emission_prob_dict[test_letter] = {}         # Nested dictionary to store emission probabilities against each training letter

    for train_letter in train_letters:

        black_dot_count = 0
        white_dot_count = 0
        non_matching_count = 0

        for i in range(0, CHARACTER_HEIGHT):
            for j in range(0, CHARACTER_WIDTH):
                if test_letters[test_letter][i][j] == train_letters[train_letter][i][j]:
                    if test_letters[test_letter][i][j] == '*':
                        black_dot_count += 1
                    else:
                        white_dot_count += 1
                else:
                    non_matching_count += 1
        
        emission_prob_dict[test_letter][train_letter] = (0.9 * black_dot_count + 0.5 * white_dot_count + 0.1 * non_matching_count)/(CHARACTER_WIDTH * CHARACTER_HEIGHT)

# Function to determine the most probable character at each position in the test image using Simplified Naive Bayes based on the 
# emission probability of the observed character
def simplified_bayes_net(emission_prob_dict):

    result_string = ''

    for test_letter in range(0, len(test_letters)):
        result_string += max(emission_prob_dict[test_letter], key = lambda x: emission_prob_dict[test_letter][x])
    
    print("Simple: " + result_string)
    
# Function to determine the most probable sequence of characters in the test image using HMM with MAP inference (Viterbi)
def viterbi(train_letters, test_letters, emission_prob_dict, initial_prob_dict, transition_prob_dict):

    final_prob_dict = {}

    # Calulating the final probability for each character in the test image using initial_probability, emission probability and transition
    # probability and then select the character with max probability
    for test_letter in range(len(test_letters)):
        final_prob_dict[test_letter] = {}

        for train_letter in train_letters:
            curr_max = -9999999

            if test_letter == 0:
                result_prob = math.log(emission_prob_dict[0][train_letter], 2) 
                final_prob_dict[test_letter][train_letter] = result_prob
            else:
                 for prev_train_letter in train_letters:
                    if train_letter in transition_prob_dict and prev_train_letter in transition_prob_dict[train_letter]:
                        result_prob = transition_prob_dict[train_letter][prev_train_letter] + final_prob_dict[test_letter - 1][prev_train_letter]
                    else:
                        result_prob = math.log(math.pow(10, -8), 2) + final_prob_dict[test_letter - 1][prev_train_letter]
                    
                    if result_prob > curr_max:
                        curr_max = result_prob
                 
                 final_prob_dict[test_letter][train_letter] = curr_max + math.log(emission_prob_dict[test_letter][train_letter], 2)

    
    result_string = ''

    for test_letter in range(len(test_letters)):
        result_string += max(final_prob_dict[test_letter], key = lambda x: final_prob_dict[test_letter][x])

    print("   HMM: " + result_string) 
        

#####
# main program
if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)


#Dictionary to store emission probabilities
emission_prob_dict = {} 

#Dictionary to store initial probabilities
initial_prob_dict = {}

#Dictionary to store transition probabilities
transition_prob_dict = {}

get_emission_prob(train_letters, test_letters, emission_prob_dict)

# print('------Emission Dictionary--------')
# print(emission_prob_dict)

simplified_bayes_net(emission_prob_dict)

get_initial_prob(train_letters, test_letters, initial_prob_dict, train_txt_fname)

# print('------Initial Dictionary--------')
# print(initial_prob_dict)

get_transition_prob(train_letters, test_letters, transition_prob_dict, train_txt_fname)

# print('------Transition Dictionary--------')
# print(transition_prob_dict)

viterbi(train_letters, test_letters, emission_prob_dict, initial_prob_dict, transition_prob_dict)

##

## Below is just some sample code to show you how the functions above work. 
# You can delete this and put your own code here!


# Each training letter is now stored as a list of characters, where black
# dots are represented by *'s and white dots are spaces. For example,
# here's what "a" looks like:
# print("\n".join([ r for r in train_letters['a'] ]))

# Same with test letters. Here's what the third letter of the test data
# looks like:
# print("\n".join([ r for r in test_letters[2] ]))



# The final two lines of your output should look something like this:
# print("Simple: " + "Sample s1mple resu1t")
# print("   HMM: " + "Sample simple result") 


