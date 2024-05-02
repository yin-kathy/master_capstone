'''
This program recodes the labels to
consolidate categories

new category
mapping
input: labeled_review_topics.csv
output: relabeled dataframe new_review_topics
document: new topic list
'''

import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

all_labels = ['academics', 'value', 'diversity', 'campus', 'location', 'dorms',
               'campus_food', 'student_life', 'athletics', 'party_scene', 'professors',
               'safety', 'irrelevant', 'administration', 'unclear', 'affordability',
               'campus_resource', 'career_prep', 'alumni_network', 'community', 'mission']

label_replace = {'alumni_network': 'student_life', 'mission': 'administration',
                 'party_scene': 'student_life', 'safety': 'campus',
                 'affordability': 'value', 'athletics': 'student_life',
                 'campus_food': 'campus', 'dorms': 'campus',
                 'career_prep': 'administration', 'community': 'student_life',
                 'location': 'campus', 'campus_resource': 'administration',
                 'diversity': 'student_life'}


def replace_labels(label_list):

    return [label_replace.get(label, label) for label in label_list]

def create_label_list(df, label_col):
    '''
    transform columns from labels stored as string separated by comma
    to a nested list
    '''

    df[label_col] = df[label_col].apply(
        lambda x: [topic.strip() for topic in x.strip('[]').split(',')])

    return df

def list_to_str(df, col_name):

    df[col_name] = df[col_name].apply(lambda x: ','.join(x))

    return df


def recode_labels(all_labels):
    changes_map = {}
    current_labels = all_labels
    while True:
        # Print all possible topics
        print("Available topics:", all_labels)

        # Get user input for the label they want to recode
        original_label = input("Enter the label you want to recode: ")
        if original_label not in all_labels:
            print("This label is not in the list of possible labels.")
            continue

        # Get the new label from the user
        new_label = input("Enter the new label for '{}': ".format(original_label))

        # Update the dictionary
        changes_map[original_label] = new_label
        current_labels.remove(original_label)

        # Ask if the user wants to continue
        continue_choice = input("Do you want to recode another label? (yes/no): ")
        if continue_choice.lower() != 'yes':
            break

    print(changes_map)

    return changes_map

def create_final_mappings(changes_map = {'value': 'dorms', 'dorms': 'campus'}):
    final_map = {}
    for original, new in changes_map.items():
        print(original, new)
        final_label = new
        while final_label in changes_map:
            print(final_label)
            final_label = changes_map[final_label]
        final_map[original] = final_label
    print(f'final map is {final_map}')
    return final_map

def main():

    df_review_topics = pd.read_csv('labeled_review_topics.csv')

    changes_map = recode_labels(all_labels)
    print(f'changes map is {changes_map}')
    final_map = create_final_mappings(changes_map)
    print(f'final map is {final_map}')
    #
    # df = create_label_list(df_review_topics, 'topic')
    # df['topic'] = df['topic'].apply(changes_map)
    # df['topic'] = df['topic'].apply(lambda x: list(set(x)))
    # df = list_to_str(df, 'topic')
    # print(df)
    # df.to_csv('new_review_topics', index=False)

# main()

# df_review_topics = pd.read_csv('labeled_review_topics.csv')
#
create_final_mappings()

