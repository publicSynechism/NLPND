# NLPND
Udacity's Natural Language Processing Nanodegree

- [Markdown Syntax](https://daringfireball.net/projects/markdown/syntax)
- [Embed Math](https://www.codecogs.com/eqnedit.php?latex=\mathcal{W}(A,f)&space;=&space;(T,\bar{f}))
- [Creating a Python Package](https://packaging.python.org/tutorials/packaging-projects/)
- [Beginner's Guide to Contributing to a GitHub Project](https://akrabat.com/the-beginners-guide-to-contributing-to-a-github-project/)
- [Contributing to a GitHub Project](https://github.com/MarcDiethelm/contributing/blob/master/README.md)

### Contents

- [Introduction to Natural Language Processing](#introduction)
- [Text Processing](#text-processing)
- [Spam Classifier with Naive Bayes](#spam-classifier-with-naive-bayes)

### Important Links

- [NLTK Documentation](https://www.nltk.org/)
- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [PDF - Speech and Language Processing])(https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf)

####

# *Introduction*

Incredible progress has been made in the field of NLP, yet we still struggle to
communicate human language intelligently to computers.

#### Structured Language

Human language lacks a precisely defined structure and this is required for computers to
interpret information. Mathematics and formal logic are examples of structured languages
outside of programming languages. Only unambiguous language is suitable for computers to understand.

#### Grammar

Computers use standard forms of expressing grammars and algorithms to parse properly formed
statements to understand exactly what is being communicated. If this form is not met,
the computer will return an error.

Backus-Naur Form (BNF) uses simple notation to specify grammar.

- [Grammars - Intro to CS](https://classroom.udacity.com/courses/cs101/lessons/48299949/concepts/487192400923)

#### Unstructured Text

Language has defined grammatical rules. Human language can still be complex and unstructured,
yet we are still able to understand each other even when the language is ambiguous.

Computers deal with unstructured text by processing words and phrases:
- Keywords
- Parts of Speech
- Named entities
- Dates & Quantities

Using this information, computers can attempt to *parse* statements to extract meaning.

Higher levels, computers can analyze documents:
- frequent and rare words
- tone and sentiment
- document clustering

#### Counting Words

Consider this passage:

> As I was waiting, a man came out of a side room, and at a glance I was sure he must be Long John. His left leg was cut off close by the hip, and under the left shoulder he carried a crutch, which he managed with wonderful dexterity, hopping about upon it like a bird. He was very tall and strong, with a face as big as a ham—plain and pale, but intelligent and smiling. Indeed, he seemed in the most cheerful spirits, whistling as he moved about among the tables, with a merry word or a slap on the shoulder for the more favoured of his guests.

-- Excerpt from Treasure Island, by Robert Louis Stevenson

```
"""Count words."""
import re
def count_words(text):
    """Count how many times each unique word occurs in text."""
    counts = dict()  # dictionary of { <word>: <count> } pairs to return

    # TODO: Convert to lowercase
    text = text.lower()
    # TODO: Split text into tokens (words), leaving out punctuation
    # (Hint: Use regex to split on non-alphanumeric characters)
    tokens = re.findall('\w+',text)
    # TODO: Aggregate word counts using a dictionary
    for i in tokens:
        counts[i] = counts.get(i, 0) + 1
    return counts


def test_run():
    with open("input.txt", "r") as f:
        text = f.read()
        counts = count_words(text)
        sorted_counts = sorted(counts.items(), key=lambda pair: pair[1], reverse=True)

        print("10 most common words:\nWord\tCount")
        for word, count in sorted_counts[:10]:
            print("{}\t{}".format(word, count))

        print("\n10 least common words:\nWord\tCount")
        for word, count in sorted_counts[-10:]:
            print("{}\t{}".format(word, count))


if __name__ == "__main__":
    test_run()
```
```
OUTPUT:
10 most common words:
Word	Count
a	9
the	6
he	6
and	5
was	4
as	4
with	3
left	2
of	2
his	2

10 least common words:
Word	Count
pale	1
whistling	1
more	1
sure	1
hopping	1
plain	1
like	1
crutch	1
leg	1
in	1
```

#### Contextual Dependence

Consider this movie review:

> "I was lured to see this on the promise of a smart, witty slice  of old fashioned
fun and intrigue - I was conned."

To us, this is clearly a negative review, yet computers often make mistakes on statements like this when they try to analyze them. (Notice all the positive words at the beginning)

But computers have an even harder time with statements like this:

> "The sofa didn't fit through the door because it was too narrow."

What does *IT* refer to?

We know that *it* refers to the door because we apply our knowledge about the physical
world, and "narrow" is a word we would use to describe the door in this situation.

Computers aren't good at figuring out these semantics.

#### NLP Pipelines

A common NLP pipeline involves:
- Text Processing
- Feature Extraction
- Modeling

This is not a linear process.

Text Processing:

We process text because the text we use for NLP is often not in a clean format.
An example of this is website HTML code.
We might also need to consume PDF, OCR, or Speech Recognition output.

We also modify capitalization, common words, and punctuation.

Feature Extraction:

Computers store text data using binary from ASCII or Unicode. Note that individual characters
don't carry meaning. Only words and statements capture meaning. This is similar to the way pixel data is aggregated to make a meaningful photo.

The way we extract features depends on the model or end goal we want to use the text for.
For example, statistical models will require numerical values.

Modeling:

- designing a model (statistical or ML)
- fitting parameters to a training data using optimization procedure
- making predictions about unseen data

Numerical features allow us to use any ML model.

How we utilize the model is up to us.

# *Text Processing*

Install Packages:
```
conda install --file requirements.txt
```
```
pip install -r requirements.txt
```

We are going to learn how to read text data from different sources and prepare it for feature extraction.

Procedure:
- Cleaning
- Normalization
- Tokenization
- Stop Word Removal
- Part-of-Speech Tagging
- Named Entity Recognition
- Stemming and Lemmatization

#### Typical Workflow

> "Jenna went back to University."

Normalize:
>"jenna went back to university"

Tokenize:
> <"jenna", "went", "back", "to", "university">

Remove Stop Words:
> <"jenna", "went", "university">

Stem / Lemmatize:
> <"jenna", "go", "univers">

# *Spam Classifier with Naive Bayes*

Uses conditional probability. Very fast and easy to train. Application to NLP.

#### Bayes Theorem Example

Alex and Brenda work in our office. Alex and Brenda are both in the office the same amount of time.

Today, we saw someone walk by really fast and we're not sure who it is.

Since we know Alex and Brenda are both in the office the same amount of time,

> P(Alex) = 0.5,
>
> P(Brenda) = 0.5

However, this person had a red sweater.

Alex wears red 2 times a week.

Brenda wears red 3 times a week.

> P(Alex) = 0.4
>
> P(Brenda) = 0.6
