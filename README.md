# chatbot
A simple chatbot trained from scratch on a dummy dataset.

This is a Supervised Model with no generative capablities.
This just gives a framework to train a simple chatbot to handle predefined scenarios.

This works pretty well with 99% accuracy.

Sometimes it faces some challenge with grammatically incorrect words.
We can solve this by implementing a Trie on the client side to check for grammatical errors.

Take a look at [[Train_Bot.json]]. The trained model will be capable of responding to those questions.
 
### Steps to run locally

_It is advisable to create a new environment for every python project._

```bash
pip install -r requirements.txt
python train.py
python run.py
```
