NLP Mini-Pipeline

Run examples:
  python nlp_pipeline.py --input source_text_short.txt --output result.tsv
  python nlp_pipeline.py --input source_text_short.txt --output result.tsv --pos manual

Optional lemmatization dataset:
  Download lemmatization-en.txt from:
  https://github.com/michmech/lemmatization-lists/blob/master/lemmatization-en.txt
  and run:
  python nlp_pipeline.py --input source_text_short.txt --output result.tsv --lemma-dataset lemmatization-en.txt
