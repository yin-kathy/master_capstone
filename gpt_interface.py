import openai
import pandas as pd
import os
from datetime import datetime
import csv

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

openai.api_key = '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'

# TOPICS_LIST = ['academics', 'value', 'diversity', 'campus', 'location', 'dorms',
#                'campus_food', 'student_life', 'athletics', 'party_scene', 'professors',
#                'safety', 'irrelevant', 'administration', 'unclear', 'affordability',
#                'campus_resource', 'career_prep', 'alumni_network', 'community', 'mission']

TOPICS_LIST = ['academics', 'campus',
                   'student_life', 'professors', 'administration']

X_train = pd.read_csv('data/X_train_test.csv')

'''
PROMPT ENGINEER
'''

system_message_template = '''You are an assistant that reviews school reviews 
and identifies the main topics mentioned.'''

delimiter = '####'
delimiter_1 = '----'


user_topic_prompt = '''Please, define the main topics in this review.'''

example_1 = '''CC is extremely homogeneous. The visible majority of the student body are white, upper class, 
heterosexual, and democrats. However, people tend to be accepting of differing race, economic background, religion, 
sexual orientation, etc. Students and professors alike claim to be open-minded, but conservatives are received 
poorly. People stick to what they know, i.e. international students tend to hang out together. It takes effort to 
break the bubble within the CC bubble.'''

output_1 = ['diversity']


example_2 = '''NYU is a fun experience if you have the means. The campus and NYC in general  is full of things to do 
and parties. However, if you're not financially well off you will have to work multiple jobs and hustle to afford 
housing, food and basic necessities. New York is very expensive. If you don't receive enough financial aid help, 
the tuition will be a large burden as well. So for students who come from a humble background, adjusting and thriving 
to this lifestyle may come at the expense of mental sanity. Having to work full time and attend school full time is 
as strenuous as it gets.'''

output_2 = ['affordability']

# messages = [
#   {'role': 'system', 'content': system_prompt},
#   {'role': 'user', 'content': user_translation_prompt},
#   {'role': 'assistant', 'content': model_translation_response},
#   {'role': 'user', 'content': user_topic_prompt}
# ]

message_tm = ''' You are a review annotator. I am giving you an online review of
colleges by students. I want you to do a topic modelling for me. 
I want to know what are the major aspects students mentioned from
what they input. Tag the topics within this list: [academics, value,
diversity, campus, location, dorms, campus_food, student_life, 
athletics, party_scene, professors, safety, administration, affordability,
campus_resource, career_prep, alumni_network, community, mission]. 
If the review is not directly mention these topics related to what students 
and family might care about when choosing a school, tag 'irrelevant'. 
If the review is too general and does not specify any specific topic, tag 'unclear'. 
The output should be strictly following this format:[the primary topic tags].
The input of the texts is {}
'''

'''
FUNCTIONS
'''


def get_model_response(messages,
                       model='gpt-3.5-turbo',
                       temperature=0,  # randomness in the model
                       max_tokens=1000):
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content


def format_prompt(review, prompt_type, system_message=system_message_template):
    '''
    pass on the individual review and return the whole prompt to pass directly to get_model_response
    function
    '''

    allowed_types = ['zero-shot', 'one-shot', 'few-shot']

    if prompt_type not in allowed_types:
        raise ValueError(f"Invalid type provided. Choose from {allowed_types}")

    if prompt_type == 'zero-shot':

        user_message = f'''
        Please, define the main topics in this review separated by {delimiter}.  The topics should be strictly within 
        this list {TOPICS_LIST}. 
        Output is a list with the following format 
        [<topic1>, <topic2>]
    
        School reviews:
        {delimiter}
        {review}
        {delimiter}
        '''
    elif prompt_type == 'one-shot':
        user_message = f'''
        Please, define the main topics in this review separated by {delimiter}.  The topics should be strictly within 
        this list {TOPICS_LIST}. 
        Output is a list with the following format 
        [<topic1>, <topic2>]
    
        School reviews:
        {delimiter}
        {review}
        {delimiter}
        
        An example of defining topics in review:
        {delimiter} 
        School review: {example_1}
        Output: {output_1}
        {delimiter}
        '''

    else:
        user_message = f'''
                Please, define the main topics in this review separated by {delimiter}.  The topics should be strictly within 
                this list {TOPICS_LIST}. 
                Output is a list with the following format 
                [<topic1>, <topic2>]

                School reviews:
                {delimiter}
                {review}
                {delimiter}

                Examples of defining topics in review:
                {delimiter} 
                School review: {example_1}
                Output: {output_1}
                {delimiter}
                {delimiter} 
                School review: {example_2}
                Output: {output_2}
                {delimiter}
                '''


    message = [
        {'role': 'system',
         'content': system_message},
        {'role': 'user',
         'content': user_message},
    ]

    return message


def get_labels(df, prompt_type):

    labels = []

    for review in df['body']:
        # print(f'input: {review}')
        message = format_prompt(review, prompt_type)
        response = get_model_response(message)
        # print(f'output: {response}')
        labels.append(response)

    df['label_list'] = labels

    return df


def create_label_list(df, label_col):
    '''
    transform columns from labels stored as string separated by comma
    to a nested list
    '''

    # for response in df['label_list']:
    #     print(response)
    #     print(response.strip('[]').split(','))

    # df['label_list'] = df[label_col].apply(
    #     lambda x: [topic.strip() for topic in x.strip('[]').split(',')])

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

    df_dummies = pd.DataFrame(columns=['guid'] + TOPICS_LIST)
    df_dummies['guid'] = df['guid']
    df_dummies = df_dummies.fillna(0).set_index('guid')

    for index, row in df.iterrows():
        guid = row['guid']
        for topic in row['label_list']:
            if topic in df_dummies.columns:
                df_dummies.at[guid, topic] = 1

    df_dummies.reset_index(inplace=True)

    for topic in TOPICS_LIST:
        df_dummies[topic] = df_dummies[topic].astype(int)
    print(f'the output df from gpt_interface.py {df_dummies.head(5)}')
    return df_dummies


prompt_strategy = ['zero-shot', 'one-shot', 'few-shot']

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

def get_df_pred(df_X, prompt_strategy):

    print('run gpt_interface.py')

    dt = datetime.now()
    ts = datetime.timestamp(dt)
    series_num = str(ts).split('.')[0]  # timestamp of the run time

    df_gen_reviews = get_labels(df_X, prompt_type=prompt_strategy)

    # df_gen_reviews.to_csv('data/sample_results_test.csv',index = False)
    # print(df_gen_reviews)
    df_gen_reviews = create_label_list(df_gen_reviews, 'label_list')
    df_pred = create_dummies(df_gen_reviews, 'label_list')

    # df_pred.to_csv('data/test_pred.csv', index = False)

    return df_pred



# main()

def test():

    df_result = pd.read_csv('data/sample_results_test.csv')

    df_result['label_list'] = df_result['label_list'].apply(lambda x: x.strip("[]").replace("'", "").split(", "))
    #
    # for ls in df_result['label_list']:
    #     print(ls)
    #     tmp = ls.strip('[]')
    #     ls_topic = tmp.split(',')
    #     print(ls_topic)

    # df_gen_reviews = create_label_list(df_gen_reviews, 'label_list')

    print(df_result)

    for ls in df_result['label_list']:
        for label in ls:
            print(label)
