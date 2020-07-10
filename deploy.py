from flask import request
from transformers import BertForQuestionAnswering
import flask
import torch
from transformers import BertTokenizer

app = flask.Flask(__name__)
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
@app.route('/')
def index():

    if request.args:

        context = request.args["context"]
        question = request.args["question"]
        
        
        

        

        answer = ans(context, question)


        return flask.render_template('index.html', question=question, answer=answer)
    else:
        return flask.render_template('index.html')

def ans (question,answer_text):
    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = tokenizer.encode(question, answer_text)

    # print('The input has a total of {:} tokens.'.format(len(input_ids)))


    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = tokenizer.encode(question, answer_text)

    # print('The input has a total of {:} tokens.'.format(len(input_ids)))
    # BERT only needs the token IDs, but for the purpose of inspecting the 
    # tokenizer's behavior, let's also get the token strings and display them.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # For each token and its id...
    for token, id in zip(tokens, input_ids):

        # If this is the [SEP] token, add some space around it to make it stand out.
        if id == tokenizer.sep_token_id:
            print('')

        # Print the token string and its ID in two columns.
    #     print('{:<12} {:>6,}'.format(token, id))

        if id == tokenizer.sep_token_id:
            print('')

    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)


    # Run our example through the model.
    start_scores, end_scores = model(torch.tensor([input_ids]), # The tokens representing our input text.
                                     token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text

    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Combine the tokens in the answer and print it out.
    answer = ' '.join(tokens[answer_start:answer_end+1])

    # print('Answer: "' + answer + '"')

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):

        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]

        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]
    return answer           
        

if __name__ == "__main__":
    app.run(debug=True, host='172.31.28.15', port=8890)
