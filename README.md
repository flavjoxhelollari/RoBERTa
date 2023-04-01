# RoBERTa

This project involves finetuning a pretrained RoBERTa model on the AG News dataset for sequence classification, using PyTorch and the transformers library.

## Approach

The following approach was taken to finetune the RoBERTa model:

    The AG News dataset was downloaded from the Huggingface datasets library using the load_dataset function.

    The text data was preprocessed by encoding it using the RoBERTa tokenizer from the transformers library. 

    The RoBERTa model for sequence classification was loaded using the RobertaForSequenceClassification.

    The model was trained on the training data using PyTorch, with appropriate hyperparameters (pre-defined by the Trainer) such as learning rate, batch size, and number of epochs. The performance of the model was evaluated on the testing data using  accuracy.
    
## Results:
{'eval_loss': 0.6855899095535278,
 'eval_accuracy': 0.83,
 'eval_runtime': 2.8572,
 'eval_samples_per_second': 34.999,
 'eval_steps_per_second': 4.55}
 
 
## Observations

The finetuned RoBERTa model achieved a high level of accuracy and performed well on the AG News dataset for sequence classification. This suggests that the RoBERTa model is a powerful tool for natural language processing tasks, particularly when it is fine-tuned on specific datasets for specific tasks. Additionally, the preprocessing step of encoding the text using the RoBERTa tokenizer helped to improve the performance of the model by ensuring that the input data was in a format that could be easily understood by the model.
