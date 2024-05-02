'''
This program recodes the labels to
consolidate categories

new category
mapping
input: labeled_review_topics.csv
output: relabeled dataframe new_review_topics
document: new topic list
'''
import csv
import pandas as pd
import os

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


def replace_labels(df, col, map):

    '''
    This function will take in a column of list in string format, and replace the original labels in the list
    to new labels based on the mapping dictionary in map
    '''

    df[col] = [','.join(map.get(label, label) for label in topics.split(',')) for topics in df['topic']]

    return df

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
        while True:

            continue_choice = input("Do you want to recode another label? (yes/no): ")
            if continue_choice.lower() in ['yes', 'no']:
                break
            else:
                print("Please enter 'yes' or 'no'.")

        if continue_choice.lower() == 'no':
            break


    print(changes_map)

    return changes_map

def create_final_mappings(changes_map):
    final_map = {}
    for original, new in changes_map.items():

        final_label = new
        while final_label in changes_map:
            final_label = changes_map[final_label]
        final_map[original] = final_label

    return final_map

def store_recoded_file(df, final_map):

    '''



    if recode_log not present:
        create csv

    else:
        get series number
        write filename: review_{series_number} and final map to the file
    '''

    log_file_path = os.path.join('labeled_data', 'recode_log.csv')

    file_exists = os.path.isfile(log_file_path)

    if not file_exists:
        id = 0
    else:
        with open(log_file_path, mode = 'r') as file:
            reader = csv.reader(file)
            next(reader, None)
            ids = [int(row[0]) for row in reader]
            id = max(ids) + 1 if ids else 0

    file_name = f'df_recoded_review_{id}.csv'
    file_path = os.path.join('labeled_data', file_name)

    df.to_csv(file_path, index_label=False)

    log_recode(final_map, file_name, id, file_path)

    print(f'dataframed strored as {file_name}')
    print(f'Log updated.')


def log_recode(final_map, file_name, batch_no, path):

    # Check if file exists
    filename = os.path.join('labeled_data', 'recode_log.csv')
    file_exists = os.path.isfile(filename)

    # Data to write
    row = [batch_no, final_map, file_name, path]

    # Open the file in append mode or write mode if it does not exist
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        # If the file does not exist, write the header first
        if not file_exists:
            writer.writerow(['batch_no', 'final_map', 'file_name', 'path'])

        # Write the data row
        writer.writerow(row)


def main():

    df = pd.read_csv('labeled_data/labeled_review_topics.csv')

    changes_map = recode_labels(all_labels)
    final_map = create_final_mappings(changes_map)

    # df = create_label_list(df, 'topic')

    df = replace_labels(df, 'topic', final_map)

    '''
    todo: store the coded scheme, file_name, and file size (row num) 
    '''

    store_recoded_file(df, final_map)

    # print(df.head(10))
    # df.to_csv('new_review_topics', index=False)

# main()
# df_review_topics = pd.read_csv('labeled_review_topics.csv')
main()

