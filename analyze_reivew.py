from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
from gpt_interface import *
from sklearn.metrics import precision_recall_fscore_support

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df_annotated = pd.read_csv('data/df_hl_topsen.csv')
df_review_topics = pd.read_csv('data/new_review_topics')
df_main = pd.read_csv('data/df_main.csv')
df_review_topics = df_review_topics[df_review_topics["topic"].str.contains("irrelevant|unclear") == False]

# Test: remove the roles that either contains "irrelevant" or "unclear"

# TOPICS_LIST_NEW = ['academics', 'value', 'diversity', 'campus', 'location', 'dorms',
#                'campus_food', 'student_life', 'athletics', 'party_scene', 'professors',
#                'safety', 'irrelevant', 'administration', 'unclear', 'affordability',
#                'campus_resource', 'career_prep', 'alumni_network', 'community', 'mission']
# TOPICS_LIST_NEW = ['academics', 'campus',
#                    'student_life', 'professors', 'irrelevant', 'administration', 'unclear']
TOPICS_LIST_NEW = ['academics', 'campus',
                   'student_life', 'professors', 'administration']

SAMPLE_SIZE = 20
RANDOM_STATE = 20

'''
Datasets:  
df_hl_topsen.csv # review body and true ABSA labels 
labeled_review_topics.csv # review ids with true topic labels 
df_main.csv # all reviews dataset (10k)
entity_id.csv # university name and entity id
 
Workflow: 
1. train-test split 
2. get labels from gpt api 
3. calculate matrix 
4. store 

'''

def get_sample(df, n, random_state):
    df_sample = df.sample(n, random_state=random_state)  # the id of samples

    X_sample = df_sample.merge(df_main, how='left', on='guid')[
        ['guid', 'body']].drop_duplicates()  # the input body of training id
    y_sample = df_sample

    return X_sample, y_sample

def data_splitter(df, test_size=0.2, val_size=0.25, random_state=42):
    '''
    This code will take the sample dataset and split it into
    train, validate, and test sets

    1. randomly select guid from the sample dataframe (df_sample)
    2. use id to get review text (df_main) and topic labels (df_review_topics)
    3. the result will return train, validate, and test with x and y_expect

    :return: df_train, df_val, df_test
    '''

    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)

    df_train, df_val = train_test_split(df_train, test_size=val_size, random_state=random_state)

    return df_train, df_val, df_test


def create_label_list(df, label_col):
    '''
    transform columns from labels stored as string separated by comma
    to a nested list
    '''

    df[label_col] = df[label_col].apply(
        lambda x: [topic.strip() for topic in x.strip('[]').split(',')])

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

    df_dummies = pd.DataFrame(columns=['guid'] + TOPICS_LIST_NEW)
    df_dummies['guid'] = df['guid']
    df_dummies = df_dummies.fillna(0).set_index('guid')

    for index, row in df.iterrows():
        guid = row['guid']
        for topic in row[col_name]:
            if topic in df_dummies.columns:
                df_dummies.at[guid, topic] = 1

    df_dummies.reset_index(inplace=True)

    return df_dummies

def get_matrix(y_expected, y_pred):

    target_names = y_expected.columns.tolist()
    y_expected = y_expected.astype(int).values
    y_pred = y_pred.astype(int).values

    output_dic = classification_report(y_expected, y_pred, output_dict=True, target_names=target_names, zero_division=1.0)

    perform_matrix = pd.DataFrame.from_dict(output_dic, orient='index')

    agg_matrix = ['micro', 'macro', 'samples', 'weighted']
    for method in agg_matrix:
        agg_stats = precision_recall_fscore_support(y_expected, y_pred, average= method)

        print(f'The {method} average is {agg_stats}')

    return perform_matrix

def new_get_matrix(y_expected, y_pred):
    # Ensure inputs are pandas DataFrames
    assert isinstance(y_expected, pd.DataFrame), "y_expected must be a pandas DataFrame"
    assert isinstance(y_pred, pd.DataFrame), "y_pred must be a pandas DataFrame"

    # Debug: Ensure column consistency
    assert (y_expected.columns == y_pred.columns).all(), "Columns of y_expected and y_pred must match exactly"

    # Convert DataFrames to binary numpy arrays (ensuring data type is correct)
    y_expected_np = y_expected.astype(int).values
    y_pred_np = y_pred.astype(int).values

    # Debug: Print data types after conversion
    print("Data type of y_expected_np:", y_expected_np.dtype)
    print("Data type of y_pred_np:", y_pred_np.dtype)

    # Attempt to run classification report
    try:
        output_dic = classification_report(y_expected_np, y_pred_np, output_dict=True,
                                           target_names=y_expected.columns.tolist())
        perform_matrix = pd.DataFrame.from_dict(output_dic, orient='index')
        return perform_matrix
    except Exception as e:
        print("Error during classification_report:", str(e))
        return None

# Example usage (Make sure to define df_y_true and df_y_pred appropriately)
# perform_matrix = get_matrix(df_y_true, df_y_pred)
# if perform_matrix is not None:
#     print(perform_matrix)


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


def store_csv(df, prefix, series_num, index):
    csv_name = '_'.join([prefix, series_num])
    file_path = os.path.join('data', csv_name)
    if index == False:
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path)
    print(f'Data is stored in {file_path}!')


def get_series_num():
    dt = datetime.now()
    ts = datetime.timestamp(dt)
    series_num = str(ts).split('.')[0]  # timestamp of the run time

    return series_num


def main():
    series_num = get_series_num()  # get the current timestamp for logging the results

    X_sample, y_sample = get_sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    X_train, X_val, X_test = data_splitter(X_sample)
    y_train, y_val, y_test = data_splitter(y_sample)

    df_gen_reviews = get_labels(X_train, message_tm)
    df_gen_reviews.to_csv('data/new_sample_results.csv', index=False)

    # df_gen_reviews = pd.read_csv('data/sample_results.csv')

    y_train = create_label_list(y_train, 'topic')
    df_gen_reviews = create_label_list(df_gen_reviews, 'label_list')

    df_expected = create_dummies(y_train, 'topic')
    df_pred = create_dummies(df_gen_reviews, 'label_list')

    eval_matix = get_matrix(df_expected.drop('guid', axis=1),
                            df_pred.drop('guid', axis=1))

    log_prompt(series_num, message_tm, len(X_train), random_state=RANDOM_STATE)

    store_csv(df_pred, 'train_pred', series_num, False)
    store_csv(eval_matix, 'train_matrix', series_num, True)


def test():

    X_sample, y_sample = get_sample(df_review_topics, n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    X_train, X_val, X_test = data_splitter(X_sample)
    y_train, y_val, y_test = data_splitter(y_sample)

    y_train = create_label_list(y_train, 'topic')
    df_expected = create_dummies(y_train, 'topic')

    # prompt_type = ['zero-shot', 'one-shot', 'few-shot']
    prompt_type = ['zero-shot']

    for strategy in prompt_type:
        series_num = get_series_num()

        # df_pred = pd.read_csv('data/new_sample_results.csv')

        df_pred = get_df_pred(X_train, prompt_strategy=strategy)
        df_pred.to_csv('data/new_sample_results.csv', index=False)

        temp_exp = df_expected.drop('guid', axis=1)
        temp_pred = df_pred.drop('guid', axis=1)

        # the result should have exactly the same labels
        # print(f'the matrix of y_true {temp_exp.head(5)}')
        # print(f'the matrix of y_pred {temp_pred.head(5)}')

        eval_matrix = get_matrix(df_expected.drop('guid', axis=1),
                                 df_pred.drop('guid', axis=1))

        print(eval_matrix)
    # log_prompt(series_num, strategy, len(X_train), random_state=RANDOM_STATE)

    # store_csv(df_pred, 'new_train_pred', series_num, False)
    # store_csv(eval_matix, 'new_train_matrix', series_num, True)


test()

