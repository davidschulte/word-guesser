# word-guesser

Iteratively guessing words until you found the right one.


![A screenshot from Contexto](contexto.png "A screenshot from Contexto")

## The Premise
[Contexto](https://contexto.me/en/) is a word guessing game that is based on word similarity. Their is a target word unknown to the user. The user makes guesses and gets feedback on how similar the guess is to the target word. Similarity is not expressed as a continuous metric, but rather in how high the guess ranks with regard to similarity to the target. 

## Considerations
To solve the game, a player has to have a similar notion of what it means for two words to be similar to each other, as the creators of the game do.
Under the hood, Contexto is using [GloVe embeddings](https://nlp.stanford.edu/projects/glove/) to measure word similarities.
Furthermore, the players need to have a roughly similar vocabulary as the game to interpret the ranking. It is of course crucial that the target word is in the player's vocabulary.
In this implementation, both the game as well as the players rely on the same vocabularies, embeddings and notions of similarity (namely cosine similarity).

## Solve Contexto automatically
Before conceptualizing about a solution, it's best to think about how a human player might play the game. They will start with one or multiple guesses that are more or less random. Once they received feedback about their scores, they will favor words that are similar to good guesses and different to bad guesses.

### The QdrantWordGuesser
The QdrantWordGuesser uses Qdrant to store its vocabulary and embeddings. To select next guesses it uses Qdrant's recommendation function. It always chooses the best previous guess as a positive example and the other guesses as negative examples.
For more details see the [Vector Similarity: Going Beyond Full-Text Search | Qdrant](https://qdrant.tech/articles/vector-similarity-beyond-search/).

### The InMemoryWordGuesser
The InMemoryWordGuesser stores its vocabulary and embeddings in memory. Therefore it can only be used for small vocabulary sizes or low embedding dimensions.
When selecting next guesses it favors words that are similar to good previous guesses and different from bad previous guesses. This is handled using the scoring threshold. For example, a scoring threshold of 0.9999 declares the best possible 0.1% of guesses as good and the others as bad guesses. In contrast to the QdrantWordGuesser, this weighting is not binary but continuously calculated with an exponential score function. The InMemoryWordGuesser further allows favoring commonly used words if supplied with frequencies.

### The HumanGuesser
You can play the game in the console and try to beat the automatic solvers.