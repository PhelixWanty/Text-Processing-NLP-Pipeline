NLP Mini-Pipeline Deliverables (Generated)

Files you can submit:
1) Code:
   - nlp_pipeline.py

2) Source + resulting text files:
   - source_text_short.txt
   - source_text_long.txt
   - result_short.tsv
   - result_long.tsv
   - result_short_kept_tokens.txt
   - result_long_kept_tokens.txt

3) "Screenshots" of final printout:
   - final_printout_short.png (generated here)
   - (You can generate your own for the long text by running the script and screenshotting the console,
      or you can reuse the TSV outputs.)

Word document (manual steps 1–5 + results):
- manual_steps_1_to_5.docx

Run examples:
  python nlp_pipeline.py --input source_text_short.txt --output result.tsv
  python nlp_pipeline.py --input source_text_short.txt --output result.tsv --pos manual

Optional lemmatization dataset:
  Download lemmatization-en.txt from:
  https://github.com/michmech/lemmatization-lists/blob/master/lemmatization-en.txt
  and run:
  python nlp_pipeline.py --input source_text_short.txt --output result.tsv --lemma-dataset lemmatization-en.txt
