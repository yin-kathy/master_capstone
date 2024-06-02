import openai
import pandas as pd
import os
from datetime import datetime
import csv

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

openai.api_key = 'MUSK' # add the API key here

TOPIC_LIST = ['academics', 'location', 'social_life', 'professors',
              'safety', 'administration', 'affordability', 'campus_facilities',
              'career_prospect', 'diversity', 'university_resources']

'''
PROMPTS
Prompt messages that will be passed to the API request 
'''

delimiter = '####'
delimiter_1 = '----'

SYSTEM_MESSAGE = '''You are an assistant that reviews school reviews 
and identifies the main topics mentioned.'''

user_topic_prompt = '''Please, define the main topics in this review.'''


'''
FUNCTIONS
'''


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



def format_prompt(review, df_example, prompt_type, system_message=SYSTEM_MESSAGE):

    """
    Constructs a formatted prompt for user input, based on a review text and specified interaction type (zero-shot, one-shot, or few-shot).
    The returned message will be passed to the get_model_response function.
    """

    user_prompt = f'''
            Please, define the main topics in the review separated by {delimiter}.  The topics should be strictly within 
            this list {TOPIC_LIST}. 
            The output should be a list with the following format 
            [<topic1>, <topic2>]'''

    allowed_types = ['zero-shot', 'one-shot', 'few-shot']

    if prompt_type not in allowed_types:
        raise ValueError(f"Invalid type provided. Choose from {allowed_types}")

    user_input = f'''
               School reviews:
                {delimiter}
                {review}
                {delimiter}

              '''

    if prompt_type == 'zero-shot':
        user_message = user_prompt + user_input
    elif prompt_type == 'one-shot':
        example_message = format_example(df_example, 1)
        user_message = user_prompt + user_input + example_message
    else:
        example_message = format_example(df_example, len(df_example))
        user_message = user_prompt + user_input + example_message

    message = [
        {'role': 'system',
         'content': system_message},
        {'role': 'user',
         'content': user_message},
    ]

    return message


def format_example(df_example, n_example):

    """
    This function constructs a formatted examples in the one-shot or few-shot prompting strategies.
    """

    msg_header = f'''An example of defining topics in review: '''

    msg_example = f''''''

    for i in range(n_example):
        body = df_example.iloc[i]['body']
        topic = df_example.iloc[i]['topic']
        example = f'''Review: {body}
                      Output: {topic}'''
        msg_example = msg_example + example

    msg_final = msg_header + msg_example

    return msg_final


def get_labels(df, example, prompt_type):

    """
    Processes a DataFrame containing reviews and assigns labels to each review based on model-generated responses.
    """
    labels = []

    for review in df['body']:
        message = format_prompt(review, example, prompt_type)
        response = get_model_response(message)
        labels.append(response)

    df['label_list'] = labels

    return df


def create_label_list(df, label_col):

    """
    transform columns from labels stored as string separated by comma to a nested list
    """

    df['label_list'] = df['label_list'].apply(lambda x: x.strip("[]").replace("'", "").split(", "))

    return df


def create_dummies(df, col_name):
    '''
    df:: the target dataframe
    col_name:: the name within the df where a list of topics is stored
    This function takes a dataframe with reviews and labels and return a
    dataframe with dummies.
    use to produce y_expected and y_pred:
    labeled_review_topics.csv -> y_expected
    df_gen_result -> y_pred
    '''
    pd.set_option('future.no_silent_downcasting', True)

    df = create_label_list(df, col_name)

    df_dummies = pd.DataFrame(columns=['guid'] + TOPIC_LIST)
    df_dummies['guid'] = df['guid']
    df_dummies = df_dummies.fillna(0).set_index('guid')

    for index, row in df.iterrows():
        guid = row['guid']
        for topic in row['label_list']:
            if topic in df_dummies.columns:
                df_dummies.at[guid, topic] = 1

    df_dummies.reset_index(inplace=True)

    for topic in TOPIC_LIST:
        df_dummies[topic] = df_dummies[topic].astype(int)
    return df_dummies



def log_prompt(series_number, prompt_message, sample_size, random_state, filename="prompt_log.csv"):
    """
    Logs the series number, current timestamp, and a prompt message to a CSV file.
    :param series_number: int - the series number of the run
    :param prompt_message: str - the prompt message to log
    :param filename: str - the name of the CSV file to write to
    """

    # Check if file exists
    filename = os.path.join('data', 'prompt_log.csv')
    file_exists = os.path.isfile(filename)

    # Data to write
    row = [series_number, prompt_message, sample_size, random_state]

    # Open the file in append mode or write mode if it does not exist
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        # If the file does not exist, write the header first
        if not file_exists:
            writer.writerow(['series_num', 'prompt_msg', 'sample_size', 'random_state'])

        # Write the data row
        writer.writerow(row)
