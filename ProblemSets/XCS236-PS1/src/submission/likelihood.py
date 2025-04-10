import torch
import torch.nn as nn

def log_likelihood(model, text):
    """
    Compute the log-likelihoods for a string `text`
    :param model: The GPT-2 model
    :param texts: A tensor of shape (1, T), where T is the length of the text
    :return: The log-likelihood. It should be a Python scalar.
        NOTE: for simplicity, you can ignore the likelihood of the first token in `text`.
    """

    with torch.no_grad():
        ## TODO:
        ##  1) Compute the logits from `model`;
        ##  2) Return the log-likelihood of the `text` string. It should be a Python scalar.
        ##      NOTE: for simplicity, you can ignore the likelihood of the first token in `text`
        ##      Hint: Checkout Pytorch softmax: https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
        ##                     Pytorch negative log-likelihood: https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
        ##                     Pytorch Cross-Entropy Loss: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        ## 
        ## The problem asks us for (positive) log likelihood, which would be equivalent to the negative of a negative log likelihood. 
        ## Cross Entropy Loss is equivalent applying LogSoftmax on an input, followed by NLLLoss. Use reduction 'sum'.
        ## 
        ## Hint: Implementation should only takes 3~7 lines of code.
        
        ### START CODE HERE ###
        # Getting the log(softmax(logits)) which can be compared against the text
        logits, _ = model(text)
        log_softmax = nn.LogSoftmax(dim=2)
        log_softmax_values = log_softmax(logits)

        #computing the loss 
        loss_fn = nn.NLLLoss(reduction='sum')
        ce_loss_fn = nn.CrossEntropyLoss(reduction='sum')

        #getting the predictions except for last token which is for text[n+1]
        pred = log_softmax_values[0][0:-1,:]

        #ignoring the first element of input text 
        target = text[0][1:]
        
        #computing the loss for the text[1:n], prediction[0:n-1]
        nll_loss = loss_fn(pred, target)
        log_likelihood = nll_loss.item() * -1

        ce_loss = ce_loss_fn(logits[0][0:-1,:], target) * -1
        return ce_loss
        
        #return log_likelihood
    
        ### END CODE HERE ###
        #raise NotImplementedError
