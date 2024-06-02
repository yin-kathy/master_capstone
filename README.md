# LLM for College Review Analysis

This project leverages Large Language Models (LLMs) for text mining, aiming to extract and analyze topic-specific sentiments expressed by online reviews. 

## Getting Started

### Data

#### /original reviews 
- *df_main.csv*: details information of each user-generated reviews, including guid (unique key), review text, authorship, date, and entity_id (link to institution)

- *entity_id.csv*: details of institutions in the review sample, including entity_id (unique key), school name, IPEDS school id (use to link to ipeds data), review counts etc. 


#### /gpt-annotated reviews 

Reviews are tagged with topics and sentiments by OpenAI GPT3.5-Turbo with zero-shot prompting. 

#### /institution data 

Selected variables from The Integrated Postsecondary Education Data System (IPEDS), the data was collected annually by the National Center for Education Statistics, a part of the 
Institute for Education Sciences within the United States Department of Education. The data is collected in 2022. 

#### validation data 

- *df_label_true.csv*: list of topic and sentiment pairs of the validation set
- *df_true_topics.csv*: list of topics for each manual annotated reviews



