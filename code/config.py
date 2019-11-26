#encoding=utf-8

class args():

    model_dir="best_model/"
    epochs=10
    data_dir="data/prep_data/"
    valid=150 

    batch_size=8
    
    hidden_size=300
    encoder_hidden_size=300
    attention_hidden_size=300
    decoder_hidden_size=300
    char_hidden_size=50
    
    label_embedding_size=50
    char_embedding_size=50

    num_layers=2
    encoder_num_layers=2
    decoder_num_layers=2

    label_size=3
    max_len=83
    max_char_len=10
    lr=0.001
    dropout=0.5
    bidirectional=True
