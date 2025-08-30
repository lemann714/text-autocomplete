from inference_lstm import inference_lstm
from inference_transformer import inference_transformer, TOKENIZER, EOS_ID, TRANSFORMER_NAME

SHOW_TEXT_LEN = 5

if __name__=="__main__":
    # Жадная генерация
    mean_rouge_lstm_greedy, lstm_greedy_texts = inference_lstm(
                                        tokenizer=TOKENIZER,
                                        eos_id=EOS_ID,
                                        use_sampling=False)

    mean_rouge_transformer_greedy, transformer_greedy_texts = inference_transformer(
                                                transformer_name=TRANSFORMER_NAME,
                                                tokenizer=TOKENIZER,
                                                eos_id=EOS_ID,
                                                use_sampling=False)


    temperature=0.8
    top_p=0.9
    mean_rouge_lstm_sampling, lstm_sampling_texts = inference_lstm(
                                            tokenizer=TOKENIZER,
                                            eos_id=EOS_ID,
                                            use_sampling=True,
                                            temperature=temperature,
                                            top_p=top_p)

    mean_rouge_transformer_sampling, transformer_sampling_texts = inference_transformer(
                                                transformer_name=TRANSFORMER_NAME,
                                                tokenizer=TOKENIZER,
                                                eos_id=EOS_ID,
                                                use_sampling=True,
                                                temperature=temperature,
                                                top_p=top_p)
    
    print(f'{mean_rouge_lstm_greedy=}')
    print(f'{mean_rouge_transformer_greedy=}')
    print(f'{mean_rouge_lstm_sampling=}')
    print(f'{mean_rouge_transformer_sampling=}')

    print('LSTM с семплированием:')
    for j in range(SHOW_TEXT_LEN):
        print(f'PROMT: {lstm_sampling_texts[j]["prompt"]}')
        print(f'PREDICTION: {lstm_sampling_texts[j]["prediction"]}\n')

    print('TRANSFORMER с семплированием:')
    for j in range(SHOW_TEXT_LEN):
        print(f'PROMT: {transformer_sampling_texts[j]["prompt"]}')
        print(f'PREDICTION: {transformer_sampling_texts[j]["prediction"]}\n')

    print('LSTM жадная генерация:')
    for i in range(SHOW_TEXT_LEN):
        print(f'PROMT: {lstm_greedy_texts[i]["prompt"]}')
        print(f'PREDICTION: {lstm_greedy_texts[i]["prediction"]}\n')

    print('TRANSFORMER жадная генерация:')
    for i in range(SHOW_TEXT_LEN):
        print(f'PROMT: {transformer_greedy_texts[i]["prompt"]}')
        print(f'PREDICTION: {transformer_greedy_texts[i]["prediction"]}\n')