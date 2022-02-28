# knowledgeGraphQA
Question answering incorporating knowledge graphs
Milestones:
28/02/2022
- Produced knowledge graph from input data
- Started testing over different tools to learn the embeddings of the graph (PyKeen, node2vec)
- Produced word embeddings from input data using a SkipGram model (A better corpus should be used when hardware is available)
- Produced QA Tensorflow model using a EncoderDecoder + attention Pattern (Should later replace attention with the anchor loss to add context)

TODO
- QAModel needs training (deploy it on googleColab)
- Write the anchorLoss function (using a custom loss is available and should be straightfoward)
- Train the Graph's embeddings
- Write the QA prediction pipeline 
