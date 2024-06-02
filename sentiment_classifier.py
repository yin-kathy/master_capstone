import numpy as np
import openai
import pandas as pd
from sklearn.metrics import confusion_matrix
import ast
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR)

openai.api_key = 'MUSK' #insert your API key here

TOPIC_LIST = ['academics', 'location', 'social_life', 'professors',
              'safety', 'administration', 'affordability', 'campus_facilities',
              'career_prospect', 'diversity', 'university_resources']

user_prompt = """Please, predict the sentiment for each of the topics mentioned in the student research. The sentiment assigned should strictly be one of these ['negative',  'neutral', 'positive'].  
                First, I will give you the review text and a list of topics labeled for this review. You should only assign sentiments to the topics already be identified. Then, you will find
                texts that are most relevant to a specific topic and then return the sentiment of that topic one at a time. The output should be in a dictionary format: {topic_name: sentiment, topic_name: sentiment}."""

'''
PROMPT ENGINEER
'''

delimiter = '####'
delimiter_1 = '----'

SYSTEM_MESSAGE = '''You are an assistant that reviews school reviews and identifies the sentiment in each topic.'''


def get_model_response(messages,
                       model='gpt-3.5-turbo',
                       temperature=0,  # randomness in the model
                       max_tokens=1000):
    """
        Retrieves a model-generated response for a given list of messages using the OpenAI API.
    """
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content


def format_prompt(review, label_list, system_message=SYSTEM_MESSAGE):
    user_input = f'''               
                {delimiter}
                School review: {review}
                Topics in the review: {label_list} 
                {delimiter}
                '''
    user_message = user_prompt + user_input

    message = [
        {'role': 'system',
         'content': system_message},
        {'role': 'user',
         'content': user_message},
    ]


    return message

def get_labels(df):
    labels = []
    total_reviews = len(df)

    for index, row in df.iterrows():
        review = row['body']
        topics = row['label_list']
        message = format_prompt(review, topics)
        response = get_model_response(message)
        labels.append(response)

        # Progress calculation and display every 20 reviews
        if (index + 1) % 20 == 0:
            percent_done = (index + 1) / total_reviews * 100
            print(f'Completed {index + 1}/{total_reviews} reviews ({percent_done:.2f}%)')

    df['label_list'] = labels

    return df

"""
similar to the multi-label classification, we will need to get two matrix of 
y_pred and y_true for analysis. 
We need to convert the predicted dictionary to a matrix, with numbers denoted 
{'negative': -1, 'neutral': 0, 'positive': 1}. 
"""

def create_empty_dummy(df):
    """
    Creates an empty DataFrame with columns based on a unique identifier 'guid' and a predefined list of topics (TOPIC_LIST).
    The 'guid' column is set as the index of the new DataFrame. This function is typically used to prepare a structure for
    populating with dummy variables or one-hot encoded data.
    """

    df_dummies = pd.DataFrame(columns=['guid'] + TOPIC_LIST)
    df_dummies['guid'] = df['guid']
    df_dummies = df_dummies.set_index('guid')

    return df_dummies

def create_dummies_from_dict(df, col_name):

    """
    :param df: df_result, containing the column from gpt responses
    :param col_name: dictionary_column
    :return: a df with -1, 0, 1 indicating the sentiment assigned to each topic
    """

    mapping_numeric = {'negative': -1, 'neutral': 0, 'positive': 1}
    df_dummies = create_empty_dummy(df)

    for index, row in df.iterrows():

        guid = row['guid']
        dict_items = row[col_name].items()

        for item in dict_items:
            topic = item[0]
            sentiment = item[1]
            df_dummies.at[guid, topic] = mapping_numeric[sentiment]

    df_dummies.reset_index(inplace=True)

    for topic in TOPIC_LIST:
        df_dummies[topic] = df_dummies[topic].astype('Int32')

    return df_dummies

def create_y_true(df_true):

    y_true = create_empty_dummy(df_true)

    mapping_numeric = {'negative': -1, 'neutral': 0, 'positive': 1}

    for index, row in df_true.iterrows():

        guid = row['guid']

        topic = row['topic']
        sentiment = row['sentiment']

        if pd.notna(sentiment) and sentiment in mapping_numeric:
            y_true.loc[guid, topic] = mapping_numeric[sentiment]

        else:
            continue

    y_true.reset_index(inplace=True)

    for topic in TOPIC_LIST:
        y_true[topic] = y_true[topic].astype('Int32')

    y_true = y_true.drop_duplicates()

    return y_true


def parse_and_clean_dict(dictionary_str):
    try:

        dictionary_str = dictionary_str.replace('\n', '').replace(' ', '')

        dictionary = ast.literal_eval(dictionary_str)
        return dictionary
    except SyntaxError as e:
        logging.error(f"Syntax error: {e} | Problematic string: {repr(dictionary_str)}")
    except ValueError as e:
        logging.error(f"Value error: {e} | Problematic string: {repr(dictionary_str)}")
    return None

def replace_labels(df, col, map_dict):

    df_copy = df.copy()

    df_copy.loc[:, 'recoded_topic'] = df_copy.loc[:,col].replace(map_dict)

    return df_copy

def recode_df_true(df, data_idx):
    """
    :param df: df_true
    :param data_idx: the index of recoded data in recode_log.csv, used to retrieve the mappings
    :return: a dataframe of recoded labels
    """

    df_recode_log = pd.read_csv('data/labeled_data/recode_log.csv')
    map_str = df_recode_log['final_map'][data_idx]
    map_dict = parse_and_clean_dict(map_str)

    df_recoded = replace_labels(df, 'topic', map_dict)

    return df_recoded

def calculate_acsa_accuracy(df_pred, df_true):

    df_pred = df_pred.drop('guid', axis = 1)
    df_true = df_true.drop('guid', axis=1)

    correct_count = 0
    total_count = len(df_true)

    for index, (true_row, pred_row) in enumerate(zip(df_true.iterrows(), df_pred.iterrows())):

        _, true_values = true_row
        _, pred_values = pred_row

        if np.array_equal(true_values.values, pred_values.values):
            correct_count += 1

    accuracy = correct_count/total_count

    return accuracy

def create_df_confusion_matrix(df_pred, df_true):

    results = []

    df_pred = df_pred.drop('guid', axis=1)
    df_true = df_true.drop('guid', axis=1)

    for pred_col, true_col in zip(df_pred, df_true):

        y_pred = df_pred[pred_col].to_numpy()
        y_true = df_true[true_col].to_numpy()

        final_pred = y_pred[(y_pred != -99) & (y_true != -99) & (y_pred != 0) & (y_true != 0)]
        final_true = y_true[(y_pred != -99) & (y_true != -99) & (y_pred != 0) & (y_true != 0)]

        cm = confusion_matrix(final_true, final_pred, labels = [1, -1])

        TP = cm[0, 0]
        FN = cm[0, 1]
        FP = cm[1, 0]
        TN = cm[1, 1]

        support = cm.sum()

        results.append([pred_col, TP, TN, FP, FN, support])

    df_cm = pd.DataFrame(results, columns=['topic', 'tp', 'tn', 'fp', 'fn', 'support'])
    print(df_cm)

    return df_cm

TINY = 0.0001

def calculate_aspect_f1_score(df):

    results = []

    for idx, rows in enumerate(df.iterrows()):

        vals = rows[1]

        TP = vals['tp']
        TN = vals['tn']
        FP = vals['fp']
        FN = vals['fn']

        precision = TP/(TP + FP + TINY)
        recall = TP / (TP + FN + TINY)
        f1_score = 2*precision*recall/(precision+recall+TINY)

        results.append([vals['topic'], precision, recall, f1_score, vals['support']])

    df_aspect_score = pd.DataFrame(results, columns=['topic', 'precision', 'recall', 'f1_score', 'support'])

    print(df_aspect_score)

    return df_aspect_score

def calculate_micro_f1_score(df):

    TP = df['tp'].sum()
    TN = df['tn'].sum()
    FP = df['fp'].sum()
    FN = df['fn'].sum()

    precision_micro =  TP/(TP + FP + TINY)
    recall_micro = TP/(TP + FN + TINY)
    f1_score_micro = 2*precision_micro*recall_micro/(precision_micro+recall_micro+TINY)

    print(f'micro_avg_precision = {precision_micro:.4f}',
          f'micro_avg_recall = {recall_micro:.4f}',
          f'micro_avg_f1 = {f1_score_micro:.4f}',
          f'The total number of predicted pair is {TP+TN+FP+FN}')

def calculate_macro_f1_score(df):

    df = df[df['support'] != 0]

    precision_macro = df['precision'].mean()
    recall_macro = df['recall'].mean()
    f1_score_macro = 2*precision_macro*recall_macro/(precision_macro+recall_macro+TINY)

    print(f'macro_avg_precision = {precision_macro:.4f}',
          f'macro_avg_recall = {recall_macro:.4f}',
          f'macro_avg_f1 = {f1_score_macro:.4f}')


def main():
    df_result = pd.read_csv('REVIEW_TOPICS_FILE') # reviews with gpt-annotated topic lists
    df_response = get_labels(df_result)
    print(df_response.head(5))
    df_response.to_csv('FILE PATH', index= False)



def test():
    df_result = pd.read_csv('data/test_data/sentiment_result.csv')

    # Apply the function to the specific column
    df_result['dictionary_column'] = df_result['label_list'].apply(parse_and_clean_dict)

    y_pred = create_dummies_from_dict(df_result, 'dictionary_column').sort_values(by='guid').reset_index(drop=True).fillna(-99) # y_pred with guid column

    df_labeled = pd.read_csv('data/df_label_true.csv')

    df_true = df_labeled[df_labeled['guid'].isin(y_pred['guid'])] # filter out irrelevant rows

    y_true = create_y_true(df_true).sort_values(by='guid').reset_index(drop=True).fillna(-99)

    acsa_accuracy = calculate_acsa_accuracy(y_pred, y_true)
    print(f'acsa accuracy is: {acsa_accuracy:.4f}')

    df_cm = create_df_confusion_matrix(y_pred, y_true)

    df_aspect_score = calculate_aspect_f1_score(df_cm)

    calculate_micro_f1_score(df_cm)

    calculate_macro_f1_score(df_aspect_score)


main()
test()


