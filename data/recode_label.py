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

df_review_topics = pd.read_csv('labeled_review_topics.csv')

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

def main():

    df = create_label_list(df_review_topics, 'topic')

    df['topic'] = df['topic'].apply(replace_labels)
    df['topic'] = df['topic'].apply(lambda x: list(set(x)))
    df = list_to_str(df, 'topic')
    df.to_csv('new_review_topics', index=False)

main()

